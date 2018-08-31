from typing import List, Optional, Tuple

import copy
import torch
from allennlp.modules.elmo import Elmo
from allennlp.modules.lstm_cell_with_projection import LstmCellWithProjection

from model.rnn import StackedLSTMCells


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
                 vocab_to_cache: List[str] = None,
                 allow_encode: bool = True,
                 allow_decode: bool = True) -> None:
        super().__init__()

        self._elmo = Elmo(options_file=options_file,
                          weight_file=weight_file,
                          num_output_representations=1,
                          requires_grad=False,
                          do_layer_norm=do_layer_norm,
                          dropout=dropout,
                          vocab_to_cache=vocab_to_cache)

        self.output_dim = self._elmo.get_output_dim()
        del self._elmo._elmo_lstm._token_embedder

        for p in self._elmo._elmo_lstm._elmo_lstm.parameters():
            p.requires_grad = requires_grad

        for p in self._elmo._elmo_lstm._word_embedding.parameters():
            p.requires_grad = requires_grad

        self._allow_encode = allow_encode
        self._allow_decode = allow_decode

    @property
    def allow_encode(self):
        return self._allow_encode

    @property
    def allow_decode(self):
        return self._allow_decode

    def get_output_dim(self):
        return self.output_dim

    def forward(self, word_inputs: torch.Tensor, return_logit=False):
        if len(word_inputs.shape) == 1:
            word_inputs = word_inputs.unsqueeze(dim=-1)
        result = self._elmo.forward(word_inputs, word_inputs)
        output, mask = result['elmo_representations'][0], result['mask']

        logit = None
        if return_logit:
            weight = self._elmo._elmo_lstm._word_embedding.weight
            embedding_weight = torch.cat((weight, weight), dim=-1)
            logit = torch.matmul(output, embedding_weight.t())

        return output, mask, logit

    def get_forward_lstm_cells(self, n_layer=1, dropout=0.0):
        forward_layers = self._elmo._elmo_lstm._elmo_lstm.forward_layers
        if not (0 < n_layer <= len(forward_layers)):
            raise ValueError('n_layer {}, len(forward_layers) {}'.format(n_layer, len(forward_layers)))

        return MultiLayerElmoLstmCells([ElmoLstmCell(copy.deepcopy(cell)) for cell in forward_layers[:n_layer]],
                                       dropout=dropout)


class ElmoLstmCell(torch.nn.Module):
    def __init__(self, cell: LstmCellWithProjection):
        super().__init__()
        self.cell = cell
        self.memory_projection = torch.nn.Linear(cell.hidden_size, cell.cell_size, bias=False)

    def forward(self, input_: torch.FloatTensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        if states is not None:
            state, memory = states
            if len(state.shape) == 2:
                state = state.unsqueeze(dim=0)
            if len(memory.shape) == 2:
                memory = memory.unsqueeze(dim=0)
            if memory.size(-1) == self.memory_projection.in_features:
                memory = self.memory_projection(memory)

            states = (state, memory)

        return self.cell(input_.unsqueeze(dim=-2), [1] * input_.size(0), states)[1]


class MultiLayerElmoLstmCells(StackedLSTMCells):
    def __init__(self, cells: List[LstmCellWithProjection], dropout=0.0):
        super().__init__(cells, dropout)

    @property
    def bidirectional(self):
        return False
