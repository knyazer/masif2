import os

import torch

from .common import convert_to_ifbo_format


class BaseModel(torch.nn.Module):
    def __init__(self, name="ifbopfn.pt"):
        super().__init__()
        self.name = name
        # get path of the parent directory of the current directory i.e. PFNs4HPO
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        name = name if ".pt" in name[-3:] else f"{name}.pt"
        self.model = torch.load(
            os.path.join(parent_dir, f"{name}"), map_location="cpu", weights_only=False
        ).to("cuda")
        self.model.eval()

    def forward(self, x_train, y_train, x_test):
        raise NotImplementedError

    @torch.no_grad()
    def predict_mean(self, x_train, y_train, x_test):
        logits = self(x_train=x_train, y_train=y_train, x_test=x_test)
        return self.model.criterion.mean(logits)

    @torch.no_grad()
    def predict_mean_variance(self, x_train, y_train, x_test):
        logits = self(x_train=x_train, y_train=y_train, x_test=x_test)
        return self.model.criterion.mean(logits), self.model.criterion.variance(logits)

    @torch.no_grad()
    def predict_quantiles(self, x_train, y_train, x_test, qs):
        logits = self(x_train=x_train, y_train=y_train, x_test=x_test)
        return torch.cat([self.model.criterion.icdf(logits, q) for q in qs], dim=1)

    @torch.no_grad()
    def nll_loss(self, x_train, y_train, x_test, y_test):
        logits = self(x_train=x_train, y_train=y_train, x_test=x_test)
        return self.model.criterion(logits, y_test)

    @torch.no_grad()
    def get_ucb(self, x_train, y_train, x_test):
        logits = self(x_train=x_train, y_train=y_train, x_test=x_test)
        return self.model.criterion.ucb(logits, best_f=None)

    @torch.no_grad()
    def get_ei(self, x_train, y_train, x_test, f_best):
        logits = self(x_train=x_train, y_train=y_train, x_test=x_test)
        return self.model.criterion.ei(logits, best_f=f_best)


class PFN_MODEL(BaseModel):
    def forward(self, x_train, y_train, x_test):
        if x_train.shape[0] == 0:
            x_test[:, 0] = 0
        elif x_train[:, 0].min() == 0:
            x_train[:, 0] += 1
            x_test[:, 0] += 1

            # reserve id=0 to curves that are not in x_train
            # set to 0 for all id in x_test[:, 0] that is not x_train[:, 0]
            x_test[:, 0] = torch.where(
                torch.isin(x_test[:, 0], x_train[:, 0]),
                x_test[:, 0],
                torch.zeros_like(x_test[:, 0]),
            )

        single_eval_pos = x_train.shape[0]
        batch_size = 2000
        n_batches = (x_test.shape[0] + batch_size - 1) // batch_size

        results = []
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, x_test.shape[0])
            x_batch = torch.cat([x_train, x_test[start:end]], dim=0).unsqueeze(1)
            y_batch = y_train.unsqueeze(1)
            result = self.model((x_batch, y_batch), single_eval_pos=single_eval_pos)
            results.append(result)

        final_result = torch.cat(results, dim=0)
        return final_result

    def eval(self, *args):
        ifbo_inp = convert_to_ifbo_format(*args)
        _logits = self.forward(*ifbo_inp)
        _logits = torch.softmax(_logits, dim=-1).detach().cpu().numpy().squeeze()
        breakpoint()
        return _logits
