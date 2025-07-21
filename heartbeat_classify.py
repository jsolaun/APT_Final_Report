import sys
import arff
import numpy as np
import torch
from types import SimpleNamespace

# Add library to path
sys.path.insert(0, 'Time-Series-Library')


def load_arff(path):
    """Load ARFF file and return X (n, series_len) and y (n,) arrays."""
    text = open(path).read()
    header, data = text.split('@data', 1)

    # verify header using liac-arff (ignoring relational constructs)
    hdr_lines = [
        l for l in header.splitlines()
        if not l.startswith('%')
        and 'relational' not in l.lower()
        and not l.lower().startswith('@end')
    ]
    hdr = '\n'.join(hdr_lines)
    attr_count = sum(
        1 for l in hdr_lines if l.lower().startswith('@attribute')
        and 'target' not in l.lower()
    )
    dummy_row = ','.join('0' for _ in range(attr_count)) + ',normal'
    arff.loads(hdr + f'\n@data\n{dummy_row}')  # verify header format

    X, y, label_map = [], [], {}
    for line in data.strip().split('\n'):
        if not line:
            continue
        feats, label = line[1:].split("'", 1)
        feats = feats.replace('\\n', ',')
        values = [float(v) for v in feats.split(',') if v]
        if label_map.get(label.strip(',')) is None:
            label_map[label.strip(',')] = len(label_map)
        X.append(values)
        y.append(label_map[label.strip(',')])

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


class Exp_Classification:
    """Minimal classifier wrapper around library models."""

    def __init__(self, model_name, input_dim, num_classes, seq_len):
        self.args = SimpleNamespace(
            task_name='classification',
            seq_len=seq_len,
            pred_len=0,
            enc_in=input_dim,
            num_class=num_classes,
            d_model=16,
            e_layers=1,
            d_layers=1,
            n_heads=2,
            d_ff=32,
            top_k=3,
            num_kernels=6,
            dropout=0.1,
            factor=1,
        )
        if model_name == 'TimesNet':
            from models.TimesNet import Model
        elif model_name == 'Crossformer':
            from models.Crossformer import Model
        else:
            raise ValueError('Unknown model')
        self.model = Model(self.args)
        self.device = torch.device('cpu')
        self.model.to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.crit = torch.nn.CrossEntropyLoss()

    def fit(self, X, y, epochs=1, batch_size=8):
        ds = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y))
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
        self.model.train()
        for _ in range(epochs):
            for bx, by in dl:
                bx = bx.to(self.device).unsqueeze(-1)
                mask = torch.ones(bx.shape[0], bx.shape[1], device=self.device)
                out = self.model(bx, mask, None, None)
                loss = self.crit(out, by)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

    def evaluate(self, X, y, batch_size=8):
        ds = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y))
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size)
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for bx, by in dl:
                bx = bx.to(self.device).unsqueeze(-1)
                mask = torch.ones(bx.shape[0], bx.shape[1], device=self.device)
                pred = self.model(bx, mask, None, None)
                pred_label = pred.argmax(dim=1)
                correct += (pred_label.cpu() == by).sum().item()
                total += by.size(0)
        return correct / total


def main():
    X_train, y_train = load_arff('Heartbeat/Heartbeat_TRAIN.arff')
    X_test, y_test = load_arff('Heartbeat/Heartbeat_TEST.arff')
    seq_len = X_train.shape[1]
    num_classes = int(y_train.max() + 1)

    tn = Exp_Classification('TimesNet', 1, num_classes, seq_len)
    tn.fit(X_train, y_train)
    acc1 = tn.evaluate(X_test, y_test)

    cf = Exp_Classification('Crossformer', 1, num_classes, seq_len)
    cf.fit(X_train, y_train)
    acc2 = cf.evaluate(X_test, y_test)

    print(f'TimesNet accuracy: {acc1*100:.2f}%')
    print(f'Crossformer accuracy: {acc2*100:.2f}%')


if __name__ == '__main__':
    main()
