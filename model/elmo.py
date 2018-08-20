from typing import List

import torch
from allennlp.modules.elmo import Elmo


class ElmoLM(torch.nn.Module):
    """
    Compute a single layer of ELMo word representations.
    """

    def __init__(self,
                 options_file: str,
                 weight_file: str,
                 requires_grad: bool,
                 do_layer_norm: bool,
                 dropout: float = 0,
                 vocab_to_cache: List[str] = None) -> None:
        super().__init__()

        self._elmo = Elmo(options_file=options_file,
                          weight_file=weight_file,
                          num_output_representations=1,
                          requires_grad=requires_grad,
                          do_layer_norm=do_layer_norm,
                          dropout=dropout,
                          vocab_to_cache=vocab_to_cache)

        self.output_dim = self._elmo.get_output_dim()
        del self._elmo._elmo_lstm._token_embedder

    def get_output_dim(self):
        return self.output_dim

    def forward(self, word_inputs: torch.Tensor):
        if len(word_inputs.shape) == 1:
            word_inputs = word_inputs.unsqueeze(dim=-1)
        result = self._elmo.forward(word_inputs, word_inputs)
        output, mask = result['elmo_representations'][0], result['mask']
        weight = self._elmo._elmo_lstm._word_embedding.weight
        embedding_weight = torch.cat((weight, weight), dim=-1)
        logit = torch.matmul(output, embedding_weight.t())
        return output, mask, logit
