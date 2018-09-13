import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from model import beam_search as bs
from model.attention import step_attention, step_attention_score, attention_aggregate
from model.rnn import lstm_encoder, MultiLayerLSTMCells
from model.util import len_mask

INIT = 1e-2


class _GenLinear(nn.Module):
    def __init__(self, context_dim, state_dim, input_dim, bias=True):
        super().__init__()
        self._v_c = nn.Parameter(torch.Tensor(context_dim))
        self._v_s = nn.Parameter(torch.Tensor(state_dim))
        self._v_i = nn.Parameter(torch.Tensor(input_dim))
        init.uniform_(self._v_c, -INIT, INIT)
        init.uniform_(self._v_s, -INIT, INIT)
        init.uniform_(self._v_i, -INIT, INIT)

        if bias:
            self._b = nn.Parameter(torch.zeros(1))
        else:
            self.regiser_module(None, '_b')

    def forward(self, context, state, input_):
        output = (torch.matmul(context, self._v_c.unsqueeze(1))
                  + torch.matmul(state, self._v_s.unsqueeze(1))
                  + torch.matmul(input_, self._v_i.unsqueeze(1)))

        if self._b is not None:
            output = output + self._b.unsqueeze(0)
        return output


class PointerGenerator(nn.Module):
    def __init__(self,
                 vocab_size,
                 emb_dim,
                 n_hidden,
                 bidirectional,
                 n_layer,
                 dropout=0.0,
                 language_model=None):
        super().__init__()
        self._encoder = _Encoder(vocab_size,
                                 emb_dim,
                                 n_hidden,
                                 bidirectional,
                                 n_layer,
                                 dropout)

        self._decoder = _Decoder(language_model,
                                 self._embedding,
                                 n_hidden,
                                 n_layer,
                                 dropout)

        self._language_model = language_model

        if language_model is not None:
            if language_model.allow_encode:
                self._attn_lm = nn.Parameter(torch.Tensor(self._language_model.get_output_dim(), n_hidden))
                init.xavier_normal_(self._attn_lm)

    def forward(self, article, art_lens, abstract, extend_art, extend_vsize):
        attention, mask, init_dec_states, lm_attention, lm_mask = self.encode(article, art_lens)

        attention_args = (attention, mask, extend_art, extend_vsize, lm_attention, lm_mask)

        logit = self._decoder(
            attention_args,
            init_dec_states, abstract
        )

        return logit

    @property
    def _embedding(self):
        return self._encoder._embedding

    def set_embedding(self, embedding: nn.Parameter):
        """embedding is the weight matrix"""
        assert self._embedding.weight.size() == embedding.size()
        self._embedding.weight.data.copy_(embedding)
        self._decoder.copy_embedding()

    def encode(self, article, art_lens):
        attention, init_dec_states = self._encoder(article, art_lens)
        mask = len_mask(art_lens, article.device).unsqueeze(-2)

        lm_mask, lm_logit, lm_attention = None, None, None
        if self._language_model is not None and self._language_model.allow_encode:
            lm_output, lm_mask = self._language_model(article)
            lm_attention = torch.matmul(lm_output, self._attn_lm)

        return attention, mask, init_dec_states, lm_attention, lm_mask

    def batch_decode(self, article, art_lens, extend_art, extend_vsize, go, eos, unk, max_len):
        """ greedy decode support batching"""
        batch_size = len(art_lens)
        vsize = self._embedding.num_embeddings
        attention, mask, init_dec_states, lm_attention, lm_mask = self.encode(article,
                                                                              art_lens)
        mask = len_mask(art_lens, article.device).unsqueeze(-2)
        attention = (attention, mask, extend_art, extend_vsize)
        tok = torch.LongTensor([go] * batch_size).to(article.device)
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

    def batched_beamsearch(self, article, art_lens,
                           extend_art, extend_vsize,
                           go, eos, unk, max_len, beam_size, diverse=1.0):
        batch_size = len(art_lens)
        vsize = self._embedding.num_embeddings
        attention, mask, init_dec_states, lm_attention, lm_mask = self.encode(article,
                                                                              art_lens)
        all_attention = (attention, mask, extend_art, extend_vsize, lm_attention, lm_mask)
        attention = all_attention
        h, c = init_dec_states
        all_beams = [bs.init_beam(go, (h[:, i, :], c[:, i, :]))
                     for i in range(batch_size)]
        finished_beams = [[] for _ in range(batch_size)]
        outputs = [None for _ in range(batch_size)]
        for t in range(max_len):
            toks = []
            all_states = []
            for beam in filter(bool, all_beams):
                token, states = bs.pack_beam(beam, article.device)
                toks.append(token)
                all_states.append(states)
            token = torch.stack(toks, dim=1)
            states = (torch.stack([h for (h, _), _ in all_states], dim=2),
                      torch.stack([c for (_, c), _ in all_states], dim=2))
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
                    (states[0][0][:, :, batch_i, :], states[0][1][:, :, batch_i, :]),
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
                    ind = torch.LongTensor(ind).to(attention.device)
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


class _Encoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 emb_dim: int,
                 n_hidden: int,
                 bidirectional: bool,
                 n_layer: int,
                 dropout: float = 0.0):
        super().__init__()
        self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self._enc_lstm = nn.LSTM(
            emb_dim, n_hidden, n_layer,
            bidirectional=bidirectional, dropout=dropout
        )
        # initial encoder LSTM states are learned parameters
        state_layer = n_layer * (2 if bidirectional else 1)
        self._init_enc_h = nn.Parameter(
            torch.Tensor(state_layer, n_hidden)
        )
        self._init_enc_c = nn.Parameter(
            torch.Tensor(state_layer, n_hidden)
        )
        init.uniform_(self._init_enc_h, -INIT, INIT)
        init.uniform_(self._init_enc_c, -INIT, INIT)

        # project encoder final states to decoder initial states
        enc_out_dim = n_hidden * (2 if bidirectional else 1)
        self._dec_h = nn.Sequential(
            nn.Linear(enc_out_dim, n_hidden),
            nn.ReLU(),
        )
        self._dec_c = copy.deepcopy(self._dec_h)

        # multiplicative attention
        self._attn_wm = nn.Parameter(torch.Tensor(enc_out_dim, n_hidden))
        init.xavier_normal_(self._attn_wm)

    def forward(self, article: torch.Tensor, art_lens: torch.Tensor = None):
        size = (
            self._init_enc_h.size(0),
            len(art_lens) if art_lens is not None else 1,
            self._init_enc_h.size(1)
        )
        init_enc_states = (
            self._init_enc_h.unsqueeze(1).expand(*size),
            self._init_enc_c.unsqueeze(1).expand(*size)
        )
        enc_art, final_states = lstm_encoder(
            article, self._enc_lstm, art_lens,
            init_enc_states, self._embedding
        )
        if self._enc_lstm.bidirectional:
            h, c = final_states
            final_states = (
                torch.cat(h.chunk(2, dim=0), dim=2),
                torch.cat(c.chunk(2, dim=0), dim=2)
            )
        init_h = torch.stack([self._dec_h(s)
                              for s in final_states[0]], dim=0)
        init_c = torch.stack([self._dec_c(s)
                              for s in final_states[1]], dim=0)
        init_dec_states = (init_h, init_c)
        attention = torch.matmul(enc_art, self._attn_wm).transpose(0, 1)
        return attention, init_dec_states


class _Decoder(nn.Module):
    def __init__(self, language_model, embedding, n_hidden, n_layer, dropout):
        super().__init__()
        emb_dim = embedding.embedding_dim
        self._embedding = embedding
        self._lstm = None
        self._attn_w = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        init.xavier_normal_(self._attn_w)

        self._input = nn.Linear(emb_dim + n_hidden, n_hidden, bias=True)
        self._output_1 = nn.Linear(2 * n_hidden, emb_dim, bias=True)
        self._output_2 = nn.Linear(*self._embedding.weight.shape[::-1], bias=True)
        self._gen = _GenLinear(n_hidden, n_hidden, 2 * emb_dim)

        self._attn_wq_lm = None
        self._attn_wm_lm = None
        if language_model is not None:
            if language_model.allow_encode:
                self._attn_wq_lm = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
                init.xavier_normal_(self._attn_wq_lm)

                if language_model.attention == 'multi-head':
                    self._attn_wm_lm = nn.Parameter(torch.Tensor(n_hidden * 2, n_hidden))
                    init.xavier_normal_(self._attn_wm_lm)

            if language_model.allow_decode:
                self._lstm = language_model.get_forward_lstm_cells(n_layer, dropout=dropout)

        if self._lstm is None:
            self._lstm = MultiLayerLSTMCells(2 * emb_dim, n_hidden, n_layer, dropout=dropout)

    def copy_embedding(self):
        self._output_2.weight.data.copy_(self._embedding.weight)

    def forward(self, attention: tuple, init_states: tuple, target: torch.Tensor):
        max_len = target.size(1)
        states = init_states
        logits = []
        for i in range(max_len):
            tok = target[:, i:i + 1]
            logit, states, _ = self._step(tok, states, attention)
            logits.append(logit)
        logit = torch.stack(logits, dim=1)
        return logit

    def decode_step(self, tok: torch.Tensor, states: tuple, attention: tuple):
        logit, states, score = self._step(tok, states, attention)
        out = torch.max(logit, dim=1, keepdim=True)[1]
        return out, states, score

    def _step(self, tok, states, attention):
        lstm_out = states[0][-1]
        if len(lstm_out.shape) == 3:
            lstm_out = lstm_out.squeeze(0)

        extend_src, extend_vsize, context, score = self._compute_attention(lstm_out=lstm_out,
                                                                           attention=attention)

        lstm_in = self._input(torch.cat(
            [self._embedding(tok).squeeze(1), context],
            dim=1
        ))

        states = self._lstm(lstm_in, states)
        lstm_out = states[0][-1]
        if len(lstm_out.shape) == 3:
            lstm_out = lstm_out.squeeze(0)

        dec_out = self._output_1(torch.cat([lstm_out, context], dim=-1))

        # extend generation prob to extended vocabulary
        p_vocab = self._compute_gen_prob(dec_out, extend_vsize)
        # compute the probabilty of each copying
        p_gen = torch.sigmoid(self._gen(context, lstm_out, lstm_in))
        # add the copy prob to existing vocab distribution
        lp = torch.log(
            (p_gen * p_vocab
             ).scatter_add(
                dim=1,
                index=extend_src[:, :score.size(-1)],
                source=score * (1 - p_gen)
            ) + 1e-8)  # numerical stability for log
        return lp, states, score

    def topk_step(self, tok, states, attention, k):
        """tok:[BB, B], states ([L, BB, B, D]*2, [BB, B, D])"""
        h, c = states
        # attention is beamable
        lstm_out = states[0][-1]
        extend_src, extend_vsize, context, score = self._compute_attention(lstm_out=lstm_out,
                                                                           attention=attention)

        # lstm is not bemable
        nl, _, _, d_h = h.size()
        nl, _, _, d_c = c.size()
        beam, batch = tok.size()

        lstm_in_beamable = self._input(torch.cat(
            [self._embedding(tok), context], dim=-1))

        lstm_in = lstm_in_beamable.contiguous().view(beam * batch, -1)
        prev_states = (h.contiguous().view(nl, -1, d_h),
                       c.contiguous().view(nl, -1, d_c))

        h, c = self._lstm(lstm_in, prev_states)

        states = (h.contiguous().view(nl, beam, batch, -1),
                  c.contiguous().view(nl, beam, batch, -1))
        lstm_out = states[0][-1]

        dec_out = self._output_1(torch.cat([lstm_out, context], dim=-1))

        # copy mechanism is not beamable
        p_vocab = self._compute_gen_prob(
            dec_out.contiguous().view(batch * beam, -1), extend_vsize)
        p_gen = torch.sigmoid(
            self._gen(context, lstm_out, lstm_in_beamable)
        ).contiguous().view(-1, 1)
        lp = torch.log(
            (p_gen * p_vocab
             ).scatter_add(
                dim=1,
                index=extend_src[:, :score.size(-1)].contiguous().view(
                    beam * batch, -1),
                source=score.contiguous().view(beam * batch, -1) * (1 - p_gen)
            ) + 1e-8).contiguous().view(beam, batch, -1)

        k_lp, k_tok = lp.topk(k=k, dim=-1)
        return k_tok, k_lp, states, score

    def _compute_gen_prob(self, dec_out, extend_vsize, eps=1e-6):
        logit = self._output_2(dec_out)
        if len(logit.shape) == 3:
            logit = logit.squeeze(0)
        bsize, vsize = logit.size()
        if extend_vsize > vsize:
            ext_logit = torch.Tensor(bsize, extend_vsize - vsize
                                     ).to(dec_out.device)
            ext_logit.fill_(eps)
            gen_logit = torch.cat([logit, ext_logit], dim=1)
        else:
            gen_logit = logit
        gen_prob = F.softmax(gen_logit, dim=-1)
        return gen_prob

    @staticmethod
    def _matmul_func(x):
        if isinstance(x, torch.Tensor):
            return torch.matmul
        else:
            return torch.mm

    def _matmul(self, x, y):
        return self._matmul_func(x)(x, y)

    def _compute_attention(self, lstm_out, attention):
        if self._attn_wm_lm is None:
            return self._compute_modulated_attention(lstm_out, attention)

        query_func = self._matmul_func(lstm_out)
        attention, attn_mask, extend_src, extend_vsize, lm_attention, lm_mask = attention

        query = query_func(lstm_out, self._attn_w).to(attention.device)
        context, score, raw_score = step_attention(query,
                                                   attention,
                                                   attention,
                                                   attn_mask,
                                                   return_raw_score=True)

        if all(x is not None for x in (lm_attention, lm_mask, self._attn_wq_lm)):
            lm_query = query_func(lstm_out, self._attn_wq_lm).to(lm_attention.device)
            lm_context, _, lm_raw_score = step_attention(lm_query,
                                                         lm_attention,
                                                         lm_attention,
                                                         lm_mask,
                                                         return_raw_score=True)
            context = torch.matmul(torch.cat([context, lm_context], dim=-1), self._attn_wm_lm)
            score = F.softmax(raw_score + lm_raw_score[:, :raw_score.size(-1)], dim=-1)

        return extend_src, extend_vsize, context, score

    def _compute_modulated_attention(self, lstm_out, attention):
        query_func = self._matmul_func(lstm_out)
        attention, attn_mask, extend_src, extend_vsize, lm_attention, lm_mask = attention

        query = query_func(lstm_out, self._attn_w)
        score, _ = step_attention_score(query,
                                        attention,
                                        attn_mask)

        if all(x is not None for x in (lm_attention, lm_mask, self._attn_wq_lm)):
            lm_query = query_func(lstm_out, self._attn_wq_lm)
            lm_score, _ = step_attention_score(lm_query,
                                               lm_attention,
                                               lm_mask)
            score = F.normalize(score * lm_score[:, :, :score.size(-1)], p=1, dim=-1)

        context = attention_aggregate(attention, score)

        return extend_src, extend_vsize, context.squeeze(-2), score.squeeze(-2)
