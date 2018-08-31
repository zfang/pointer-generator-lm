import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from model import beam_search as bs
from model.attention import step_attention
from model.summ import Seq2SeqSumm, AttentionalLSTMDecoder
from model.util import len_mask, get_device

INIT = 1e-2


class _CopyLinear(nn.Module):
    def __init__(self, context_dim, state_dim, input_dim, lm_context_dim=None, bias=True):
        super().__init__()
        self._v_c = nn.Parameter(torch.Tensor(context_dim))
        self._v_s = nn.Parameter(torch.Tensor(state_dim))
        self._v_i = nn.Parameter(torch.Tensor(input_dim))
        init.uniform_(self._v_c, -INIT, INIT)
        init.uniform_(self._v_s, -INIT, INIT)
        init.uniform_(self._v_i, -INIT, INIT)

        self._v_l = None
        if lm_context_dim is not None:
            self._v_l = nn.Parameter(torch.Tensor(lm_context_dim))
            init.uniform_(self._v_l, -INIT, INIT)

        if bias:
            self._b = nn.Parameter(torch.zeros(1))
        else:
            self.regiser_module(None, '_b')

    def forward(self, context, state, input_, lm_context=None):
        output = (torch.matmul(context, self._v_c.unsqueeze(1))
                  + torch.matmul(state, self._v_s.unsqueeze(1))
                  + torch.matmul(input_, self._v_i.unsqueeze(1)))

        if lm_context is not None and self._v_l is not None:
            output += torch.matmul(lm_context, self._v_l.unsqueeze(1))

        if self._b is not None:
            output = output + self._b.unsqueeze(0)
        return output


class CopySumm(Seq2SeqSumm):
    def __init__(self,
                 vocab_size,
                 emb_dim,
                 n_hidden,
                 bidirectional,
                 n_layer,
                 dropout=0.0,
                 language_model=None):
        super().__init__(vocab_size,
                         emb_dim,
                         n_hidden,
                         bidirectional,
                         n_layer,
                         dropout)
        self._copy = _CopyLinear(n_hidden,
                                 n_hidden,
                                 2 * emb_dim,
                                 None if language_model is None else language_model.get_output_dim() // 2)

        self._language_model = language_model
        self._attn_wq_lm = None
        self._attn_wm_lm = None
        if language_model is not None:
            self._attn_lm = nn.Parameter(torch.Tensor(self._language_model.get_output_dim(), n_hidden))
            init.xavier_normal_(self._attn_lm)

            self._attn_wq_lm = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
            init.xavier_normal_(self._attn_wq_lm)

            self._attn_wm_lm = nn.Parameter(torch.Tensor(n_hidden * 2, n_hidden))
            init.xavier_normal_(self._attn_wm_lm)

            self._dec_lstm = language_model.get_forward_lstm_cells(n_layer, dropout=dropout)

        self._decoder = CopyLSTMDecoder(self._copy,
                                        self._attn_wq_lm,
                                        self._attn_wm_lm,
                                        self._embedding,
                                        self._dec_lstm,
                                        self._attn_wq,
                                        self._projection)

    def forward(self, article, art_lens, abstract, extend_art, extend_vsize):
        attention, mask, init_dec_states, lm_attention, lm_mask, lm_logit = self.encode(article,
                                                                                        art_lens,
                                                                                        compute_lm_logit=True)

        attention_args = (attention, mask, extend_art, extend_vsize, lm_attention, lm_mask)

        logit = self._decoder(
            attention_args,
            init_dec_states, abstract
        )

        result = {'logit': logit}
        if lm_logit is not None:
            result['lm'] = (article, lm_logit)
        return result

    def encode(self, article, art_lens, compute_lm_logit=False):
        attention, init_dec_states = super().encode(article, art_lens)
        mask = len_mask(art_lens, get_device()).unsqueeze(-2)

        lm_mask, lm_logit, lm_attention = None, None, None
        if self._language_model is not None:
            lm_output, lm_mask, lm_logit = self._language_model(article, return_logit=compute_lm_logit)
            lm_attention = torch.matmul(lm_output, self._attn_lm)

        return attention, mask, init_dec_states, lm_attention, lm_mask, lm_logit

    def batch_decode(self, article, art_lens, extend_art, extend_vsize, go, eos, unk, max_len):
        """ greedy decode support batching"""
        batch_size = len(art_lens)
        vsize = self._embedding.num_embeddings
        attention, init_dec_states = self.encode(article, art_lens)
        mask = len_mask(art_lens, get_device()).unsqueeze(-2)
        attention = (attention, mask, extend_art, extend_vsize)
        tok = torch.LongTensor([go] * batch_size).to(get_device())
        outputs = []
        attns = []
        states = init_dec_states
        for i in range(max_len):
            tok, states, attn_score = self._decoder.decode_step(
                tok, states, attention)
            attns.append(attn_score)
            outputs.append(tok[:, 0].clone())
            tok.masked_fill_(tok >= vsize, unk)
        return outputs, attns

    def decode(self, article, extend_art, extend_vsize, go, eos, unk, max_len):
        vsize = self._embedding.num_embeddings
        attention, init_dec_states = self.encode(article)
        attention = (attention, None, extend_art, extend_vsize)
        tok = torch.LongTensor([go]).to(get_device())
        outputs = []
        attns = []
        states = init_dec_states
        for i in range(max_len):
            tok, states, attn_score = self._decoder.decode_step(
                tok, states, attention)
            if tok[0, 0].item() == eos:
                break
            outputs.append(tok[0, 0].item())
            attns.append(attn_score.squeeze(0))
            if tok[0, 0].item() >= vsize:
                tok[0, 0] = unk
        return outputs, attns

    def batched_beamsearch(self, article, art_lens,
                           extend_art, extend_vsize,
                           go, eos, unk, max_len, beam_size, diverse=1.0):
        batch_size = len(art_lens)
        vsize = self._embedding.num_embeddings
        attention, mask, init_dec_states, lm_attention, lm_mask, _ = self.encode(article,
                                                                                 art_lens,
                                                                                 compute_lm_logit=False)
        all_attention = (attention, mask, extend_art, extend_vsize, lm_attention, lm_mask)
        attention = all_attention
        (h, c), prev = init_dec_states
        all_beams = [bs.init_beam(go, (h[:, i, :], c[:, i, :], prev[i]))
                     for i in range(batch_size)]
        finished_beams = [[] for _ in range(batch_size)]
        outputs = [None for _ in range(batch_size)]
        for t in range(max_len):
            toks = []
            all_states = []
            for beam in filter(bool, all_beams):
                token, states = bs.pack_beam(beam, get_device())
                toks.append(token)
                all_states.append(states)
            token = torch.stack(toks, dim=1)
            states = ((torch.stack([h for (h, _), _ in all_states], dim=2),
                       torch.stack([c for (_, c), _ in all_states], dim=2)),
                      torch.stack([prev for _, prev in all_states], dim=1))
            token.masked_fill_(token >= vsize, unk)

            topk, lp, states, attn_score = self._decoder.topk_step(
                token, states, attention, beam_size)

            batch_i = 0
            for i, (beam, finished) in enumerate(zip(all_beams,
                                                     finished_beams)):
                if not beam:
                    continue
                finished, new_beam = bs.next_search_beam(
                    beam, beam_size, finished, eos,
                    topk[:, batch_i, :], lp[:, batch_i, :],
                    (states[0][0][:, :, batch_i, :],
                     states[0][1][:, :, batch_i, :],
                     states[1][:, batch_i, :]),
                    attn_score[:, batch_i, :],
                    diverse
                )
                batch_i += 1
                if len(finished) >= beam_size:
                    all_beams[i] = []
                    outputs[i] = finished[:beam_size]
                    # exclude finished inputs
                    attention, mask, extend_art, extend_vsize, lm_attention, lm_mask = all_attention
                    masks = [mask[j] for j, o in enumerate(outputs) if o is None]
                    ind = [j for j, o in enumerate(outputs) if o is None]
                    ind = torch.LongTensor(ind).to(get_device())
                    if lm_attention is not None:
                        attention, extend_art, lm_attention = map(
                            lambda v: v.index_select(dim=0, index=ind),
                            [attention, extend_art, lm_attention]
                        )
                    else:
                        attention, extend_art = map(
                            lambda v: v.index_select(dim=0, index=ind),
                            [attention, extend_art]
                        )
                    if masks:
                        mask = torch.stack(masks, dim=0)
                    else:
                        mask = None
                    attention = (attention, mask, extend_art, extend_vsize, lm_attention, lm_mask)
                else:
                    all_beams[i] = new_beam
                    finished_beams[i] = finished
            if all(outputs):
                break
        else:
            for i, (o, f, b) in enumerate(zip(outputs,
                                              finished_beams, all_beams)):
                if o is None:
                    outputs[i] = (f + b)[:beam_size]
        return outputs


class CopyLSTMDecoder(AttentionalLSTMDecoder):
    def __init__(self, copy, attn_wq_lm, attn_wm_lm, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._copy = copy
        self._attn_wq_lm = attn_wq_lm
        self._attn_wm_lm = attn_wm_lm

    def _step(self, tok, states, attention):
        prev_states, prev_out = states
        lstm_in = torch.cat(
            [self._embedding(tok).squeeze(1), prev_out],
            dim=1
        )
        states = self._lstm(lstm_in, prev_states)
        lstm_out = states[0][-1]

        if len(lstm_out.shape) == 3:
            lstm_out = lstm_out.squeeze(0)

        extend_src, extend_vsize, context, score, lm_context = self.compute_attention(lstm_out=lstm_out,
                                                                                      attention=attention)

        if lm_context is not None and self._attn_wm_lm is not None:
            projection_context = torch.matmul(torch.cat([context, lm_context], dim=-1), self._attn_wm_lm)
        else:
            projection_context = context
        dec_out = self._projection(torch.cat([lstm_out, projection_context], dim=-1))

        # extend generation prob to extended vocabulary
        gen_prob = self._compute_gen_prob(dec_out, extend_vsize)
        # compute the probabilty of each copying
        copy_prob = torch.sigmoid(self._copy(context, lstm_out, lstm_in, lm_context))
        # add the copy prob to existing vocab distribution
        lp = torch.log(
            ((-copy_prob + 1) * gen_prob
             ).scatter_add(
                dim=1,
                index=extend_src.expand_as(score),
                source=score * copy_prob
            ) + 1e-8)  # numerical stability for log
        return lp, (states, dec_out), score

    def topk_step(self, tok, states, attention, k):
        """tok:[BB, B], states ([L, BB, B, D]*2, [BB, B, D])"""
        (h, c), prev_out = states

        # lstm is not bemable
        nl, _, _, d = h.size()
        beam, batch = tok.size()
        lstm_in_beamable = torch.cat(
            [self._embedding(tok), prev_out], dim=-1)
        lstm_in = lstm_in_beamable.contiguous().view(beam * batch, -1)
        prev_states = (h.contiguous().view(nl, -1, d),
                       c.contiguous().view(nl, -1, d))
        h, c = self._lstm(lstm_in, prev_states)
        states = (h.contiguous().view(nl, beam, batch, -1),
                  c.contiguous().view(nl, beam, batch, -1))
        lstm_out = states[0][-1]

        if len(lstm_out.shape) == 3:
            lstm_out = lstm_out.squeeze(0)

        # attention is beamable
        extend_src, extend_vsize, context, score, lm_context = self.compute_attention(lstm_out=lstm_out,
                                                                                      attention=attention)

        if lm_context is not None and self._attn_wm_lm is not None:
            projection_context = torch.matmul(torch.cat([context, lm_context], dim=-1), self._attn_wm_lm)
        else:
            projection_context = context
        dec_out = self._projection(torch.cat([lstm_out, projection_context], dim=-1))

        # copy mechanism is not beamable
        gen_prob = self._compute_gen_prob(
            dec_out.contiguous().view(batch * beam, -1), extend_vsize)
        copy_prob = torch.sigmoid(
            self._copy(context, lstm_out, lstm_in_beamable, lm_context)
        ).contiguous().view(-1, 1)
        lp = torch.log(
            ((-copy_prob + 1) * gen_prob
             ).scatter_add(
                dim=1,
                index=extend_src.expand_as(score).contiguous().view(
                    beam * batch, -1),
                source=score.contiguous().view(beam * batch, -1) * copy_prob
            ) + 1e-8).contiguous().view(beam, batch, -1)

        k_lp, k_tok = lp.topk(k=k, dim=-1)
        return k_tok, k_lp, (states, dec_out), score

    def _compute_gen_prob(self, dec_out, extend_vsize, eps=1e-6):
        embedding_weight = self._embedding.weight.t()
        if dec_out.device != embedding_weight.device:
            embedding_weight = embedding_weight.to(dec_out.device)

        logit = self._matmul(dec_out, embedding_weight)
        if len(logit.shape) == 3:
            logit = logit.squeeze(0)
        bsize, vsize = logit.size()
        if extend_vsize > vsize:
            ext_logit = torch.Tensor(bsize, extend_vsize - vsize
                                     ).to(get_device())
            ext_logit.fill_(eps)
            gen_logit = torch.cat([logit, ext_logit], dim=1)
        else:
            gen_logit = logit
        gen_prob = F.softmax(gen_logit, dim=-1)
        return gen_prob

    def _compute_copy_activation(self, context, state, input_, score):
        copy = self._copy(context, state, input_) * score
        return copy

    @staticmethod
    def _matmul_func(x):
        if isinstance(x, torch.Tensor):
            return torch.matmul
        else:
            return torch.mm

    def _matmul(self, x, y):
        return self._matmul_func(x)(x, y)

    def compute_attention(self, lstm_out, attention):
        query_func = self._matmul_func(lstm_out)
        query = query_func(lstm_out, self._attn_w)
        attention, attn_mask, extend_src, extend_vsize, lm_attention, lm_mask = attention
        context, score, raw_score = step_attention(query,
                                                   attention,
                                                   attention,
                                                   attn_mask,
                                                   return_raw_score=True)

        lm_context = None
        if all(x is not None for x in (lm_attention, lm_mask, self._attn_wq_lm)):
            lm_query = query_func(lstm_out, self._attn_wq_lm)
            lm_context, _, lm_raw_score = step_attention(lm_query,
                                                         lm_attention,
                                                         lm_attention,
                                                         lm_mask,
                                                         return_raw_score=True)
            score = F.softmax(raw_score + lm_raw_score, dim=-1)

        return extend_src, extend_vsize, context, score, lm_context
