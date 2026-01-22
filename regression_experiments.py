# SPDX-FileCopyrightText: 2026 DLR e.V.
#
# SPDX-License-Identifier: MIT

from typing import Literal
import torch
import tyro
from nac_uncertainty_regression.nac import NACWrapper, NACMode
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, d2_absolute_error_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from ucimlrepo import fetch_ucirepo 
from scipy.stats import spearmanr


plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": "\\usepackage{amsmath}\\usepackage{amssymb}",
    "savefig.bbox": "tight"
})


MODE = "SELU"

def prepare_data(X: pd.DataFrame, y: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=True)
    sc_labels = StandardScaler()
    X_train, y_train = X_train.to_numpy(), sc_labels.fit_transform(y_train.to_numpy()) # type: ignore
    X_val, y_val = X_val.to_numpy(), y_val.to_numpy()   # no scaling here!
    X_test, y_test = X_test.to_numpy(), sc_labels.transform(y_test.to_numpy()) # type: ignore

    dl_train = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_train),
            torch.tensor(y_train)
        ),
        shuffle=True,
        batch_size=256
    )
    d_val = torch.utils.data.ConcatDataset([
        torch.utils.data.TensorDataset(
            torch.tensor(X_val),
            torch.tensor(y_val)
        ),
       torch.utils.data.TensorDataset(
            torch.tensor(generate_ood(X_val.copy())),
            -torch.ones_like(torch.from_numpy(y_val)) * 1337
        ),
    ])
    dl_val = torch.utils.data.DataLoader(
        d_val,
        shuffle=True,
        batch_size=256
    )
    dl_test = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_test),
            torch.tensor(y_test)
        ),
        shuffle=True,
        batch_size=256
    )
    X_ood = generate_ood(X_test.copy())             
    dl_ood = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_ood),
            torch.tensor(y_test)
        ),
        shuffle=True,
        batch_size=256
    )
    return dl_train, dl_val, dl_test, dl_ood, sc_labels.inverse_transform

def generate_ood(data: np.ndarray):
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    # TODO changed for easier description in paper
    # data = np.random.normal(loc=np.random.choice([1, -1]) * (mean + 2 * std), scale=std/2, size=data.shape) + data
    data += np.random.normal(loc=mean + 10 * std, scale=std/2, size=data.shape)
    return data.astype(np.float32)

def setup_abalone():
    # fetch dataset 
    abalone = fetch_ucirepo(id=1) 
    # data (as pandas dataframes) 
    X = abalone.data.features  # type: ignore
    X = pd.get_dummies(X, columns=["Sex"]).astype(np.float32)
    y = abalone.data.targets.astype(np.float32) # type: ignore
    return prepare_data(X, y)
    
def setup_obesity():
    # fetch dataset 
    data = fetch_ucirepo(id=544) 
    # data (as pandas dataframes) 
    X = pd.get_dummies(data.data.features, columns=["Gender", "CAEC", "CALC", "MTRANS"]) # type: ignore
    X = pd.get_dummies(X, columns=["family_history_with_overweight", "FAVC", "SMOKE", "SCC"], drop_first=True).astype(np.float32)
    y = data.data.targets # type: ignore
    y["NObeyesdad"] = y["NObeyesdad"].map({
        "Insufficient_Weight": -1, 
        "Normal_Weight": 0, 
        "Overweight_Level_I": 1, 
        "Overweight_Level_II": 2, 
        "Obesity_Type_I": 3, 
        "Obesity_Type_II": 4,
        "Obesity_Type_III": 5
    }).astype(np.float32)
    return prepare_data(X, y)

def setup_bikeshare():
    # fetch dataset 
    data = fetch_ucirepo(id=275) 
    # data (as pandas dataframes) 
    X = data.data.features # type: ignore
    X.drop(["dteday"], axis=1, inplace=True)
    X = X.astype(np.float32)
    y = data.data.targets.astype(np.float32) # type: ignore
    return prepare_data(X, y)


def setup_wine():
    # fetch dataset 
    data = fetch_ucirepo(id=186) 
    # data (as pandas dataframes) 
    X = data.data.features.astype(np.float32) # type: ignore
    y = data.data.targets.astype(np.float32) # type: ignore
    return prepare_data(X, y)

def setup_forest():
    # fetch dataset 
    data = fetch_ucirepo(id=162) 
    X = data.data.features # type: ignore
    X = pd.get_dummies(X, columns=["month", "day"]).astype(np.float32)
    y = data.data.targets.astype(np.float32) # type: ignore
    return prepare_data(X, y)

def setup_real_estate():
    # fetch dataset 
    data = fetch_ucirepo(id=477) 
    X = data.data.features.astype(np.float32) # type: ignore
    y = data.data.targets.astype(np.float32) # type: ignore
    return prepare_data(X, y)

def setup_concrete():
    # fetch dataset 
    data = fetch_ucirepo(id=165) 
    X = data.data.features.astype(np.float32) # type: ignore
    y = data.data.targets.astype(np.float32) # type: ignore
    return prepare_data(X, y)

def setup_liver():
    # fetch dataset 
    data = fetch_ucirepo(id=60) 
    X = data.data.features.astype(np.float32) # type: ignore
    y = data.data.targets.astype(np.float32) # type: ignore
    return prepare_data(X, y)

def setup_solar_flare():
    # fetch dataset 
    data = fetch_ucirepo(id=89) 
    X = pd.get_dummies(data.data.features,  # type: ignore
        columns=["modified Zurich class", "largest spot size", "spot distribution"]
        ).astype(np.float32)
    y = data.data.targets.astype(np.float32) # type: ignore
    return prepare_data(X, y)

def setup_grid():
    # fetch dataset 
    data = fetch_ucirepo(id=471) 
    X = data.data.features.astype(np.float32) # type: ignore
    y = data.data.targets[["stab"]].astype(np.float32) # type: ignore
    return prepare_data(X, y)

def setup_conductivity():
    # fetch dataset 
    data = fetch_ucirepo(id=464) 
    X = data.data.features.astype(np.float32) # type: ignore
    y = data.data.targets.astype(np.float32) # type: ignore
    return prepare_data(X, y)


class MCDropout(torch.nn.Dropout):
    # simply turn off train and eval, so dropout is always active
    def train(self, mode: bool = True):
        super().train()
        return self
    def eval(self):
        super().train()
        return self

def get_model(activation: str, uncertainty_technique, n_output: int = 1):
    act = torch.nn.SELU if activation == "SELU" else torch.nn.ReLU
    model_fn = lambda: torch.nn.Sequential(
            torch.nn.LazyLinear(128),
            act(),
            MCDropout(p=0.3 if uncertainty_technique == "mcdropout" else 0.0),
            torch.nn.LazyLinear(128),
            act(),
            MCDropout(p=0.3 if uncertainty_technique == "mcdropout" else 0.0),
            torch.nn.LazyLinear(n_output)
        )

    if uncertainty_technique != "ensemble":
        model = model_fn()
    else:
        model = EnsembleWrapper(
            [model_fn() for _ in range(10)]
        )
    # Define optimizer, scheduler, criterion for both models.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = torch.nn.MSELoss()
    return model.train(), optimizer, criterion


class EnsembleWrapper(torch.nn.Module):
    def __init__(self, models: list[torch.nn.Module]):
        super().__init__()
        self.models = torch.nn.ParameterList(models)
        self.drop_idx = 0

    def forward(self, x) -> torch.Tensor | dict[str, torch.Tensor]:
        results = []
        for i, model in enumerate(self.models):
            if self.training and i == self.drop_idx:
                continue
            results.append(model(x))

        self.drop_idx = (self.drop_idx + 1) % len(self.models)

        results = torch.stack(results)
        if self.training:
            return torch.mean(results, dim=0)

        else:
            return dict(
                out=torch.mean(results, dim=0),
                uncertainty=torch.std(results, dim=0).squeeze(dim=1)
            )    


def train_model(data_loader, model, optimizer, criterion, inv_label_fn):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for batch in data_loader:
        data, labels = batch

        # Zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(data)
        loss = criterion(outputs.squeeze(dim=1), labels.squeeze(dim=1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        all_preds += outputs.detach().cpu().tolist()
        all_labels += labels.detach().cpu().tolist()

    return total_loss / len(data_loader.dataset), \
            mean_absolute_percentage_error(inv_label_fn(all_labels), inv_label_fn(all_preds)), \
            d2_absolute_error_score(inv_label_fn(all_labels), inv_label_fn(all_preds))
            

def evaluate_model(data_loader, model, criterion, inv_label_fn):
    model.eval()
    total_loss = 0
    num_batches = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            data, labels = batch
            outputs = model(data)
            if isinstance(outputs, dict):
                outputs = outputs["out"]
            loss = criterion(outputs.squeeze(dim=1), labels.squeeze(dim=1))

            total_loss += loss.item() * len(labels)
            num_batches += 1
            all_preds += outputs.detach().cpu().tolist()
            all_labels += labels.detach().cpu().tolist()

    return total_loss / len(data_loader.dataset), \
            mean_absolute_percentage_error(inv_label_fn(all_labels), inv_label_fn(all_preds)), \
            d2_absolute_error_score(inv_label_fn(all_labels), inv_label_fn(all_preds))


class MCWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, n_passes: int = 10) -> None:
        super().__init__()
        self.model = model
        self.n_passes = n_passes

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        model_outputs = torch.stack([
            self.model(x) for _ in range(self.n_passes)
        ])
        mean_softmax = torch.mean(model_outputs, dim=0)
        uncertainty = torch.std(model_outputs, dim=0)
        pred = torch.argmax(mean_softmax, dim=-1)
        return dict(out=mean_softmax, uncertainty=uncertainty[torch.arange(len(uncertainty)), pred])


def evaluate_uncertainty_epistemic(model: NACWrapper | MCWrapper | EnsembleWrapper, dl_val: torch.utils.data.DataLoader, dl_ood: torch.utils.data.DataLoader):    
    model.eval()        # from now, we get uncertainty estimates
    uncertainty_scores_known = []
    uncertainty_scores_unknown = []

    for X, labels in dl_val:
        out = model(X)
        uncertainty_scores_known += out["uncertainty"].tolist()

    for X, labels in dl_ood:
        out = model(X)
        uncertainty_scores_unknown += out["uncertainty"].tolist()

    correlation_dset = np.concat(
        [
            np.stack(
                [np.array(uncertainty_scores_known), np.zeros(len(uncertainty_scores_known))],
                axis=1
            ),
            np.stack(
                [np.array(uncertainty_scores_unknown), np.ones(len(uncertainty_scores_unknown))],
                axis=1
            )
        ],
        axis=0
    )
    # correlation = np.corrcoef(
    #     correlation_dset[:, 0],
    #     y=correlation_dset[:, 1],
    #     rowvar=False
    # )[0, 1]
    correlation = spearmanr(correlation_dset[:, 0], correlation_dset[:, 1]).statistic # type: ignore
    print(f"Correlation is {correlation}")

def evaluate_uncertainty_aleatoric(model: NACWrapper | MCWrapper | EnsembleWrapper, dl_val: torch.utils.data.DataLoader):
    model.eval()        # from now, we get uncertainty estimates
    uncertainty_scores = []
    mses = []

    for X, labels in dl_val:
        out = model(X)
        uncertainty_scores += out["uncertainty"].tolist()
        mses += (out["out"] - labels).square().tolist()

    correlation_dset = np.stack(
                [np.array(uncertainty_scores).squeeze(), np.array(mses).squeeze()],
                axis=1
            )
    correlation = np.corrcoef(
        correlation_dset[:, 0],
        y=correlation_dset[:, 1],
        rowvar=False
    )[0, 1]
    print(f"Correlation is {correlation}")


def param_sweep_nac(model_: torch.nn.Module, dl_train: torch.utils.data.DataLoader, dl_val: torch.utils.data.DataLoader) -> NACWrapper:
    print("Starting param sweep....")
    best_model = None
    best_corr = - float("inf")
    for layers in [["0"], ["0", "3"], ["3", "6"], ["0", "3", "6"]]:
        for O in [10, 50, 100, 1000]:
            for alpha in [0.1, 1, 10, 100, 1000]:
                for M in [20, 50, 100]:
                    model = NACWrapper(model_, layers, O=O, M=M, alpha=alpha, mode=NACMode.REGRESSION)
                    model.train()
                    for (X, lbl) in dl_train:
                        _ = model(X)

                    model.eval()
                    uncertainties, ood_marker = [], []
                    for (X, lbl) in dl_val:
                        uncertainties += model(X)["uncertainty"].tolist()
                        ood_marker += (lbl == -1337).float().tolist()

                    correlation_dset = np.stack(
                                [np.array(uncertainties)[:, None], np.array(ood_marker)],
                                axis=1
                            )
                    correlation = np.corrcoef(
                        correlation_dset[:, 0],
                        y=correlation_dset[:, 1],
                        rowvar=False
                    )[0, 1]
                    if np.isnan(correlation):
                        continue
                    if correlation > best_corr:
                        best_corr = correlation
                        best_model = model

    print("Finished param sweep!")
    assert best_model is not None   
    return best_model
                    

def main(dataset_name: Literal["wine", "abalone", "bikeshare", "obesity", "forest", 
                               "realestate", "concrete", "liver", "grid", 
                               "conductivity"] = "wine", 
         uncertainty_technique: Literal["nac", "mcdropout", "ensemble"] = "nac", 
         uncertainty_kind: Literal["epistemic", "aleatoric"] = "epistemic",
         activation: Literal["relu", "selu"] = "relu",
         epochs=1000, seed=0):

    torch.manual_seed(seed)
    np.random.seed(0)

    match dataset_name:
        case "wine": setup_fn = setup_wine
        case "abalone": setup_fn = setup_abalone
        case "bikeshare": setup_fn = setup_bikeshare
        case "obesity": setup_fn = setup_obesity
        case "forest": setup_fn = setup_forest
        case "realestate": setup_fn = setup_real_estate
        case "concrete": setup_fn = setup_concrete
        case "liver": setup_fn = setup_liver
        case "solarflare": setup_fn = setup_solar_flare
        case "grid": setup_fn = setup_grid
        case "conductivity": setup_fn = setup_conductivity
        case _: raise NotImplementedError()

    dl_train, dl_val, dl_test, dl_ood, inv_label_fn = setup_fn()
    model, optimizer, criterion = get_model(activation=activation, uncertainty_technique=uncertainty_technique, n_output=dl_train.dataset[0][1].shape[0])

    for epoch in range(epochs):
        # Train both models using the train dataset.
        loss, perc_error, d2_score = train_model(dl_train, model, optimizer, criterion, inv_label_fn)

        # Evaluate on validation dataset
        valid_loss, perc_error_val, d2_score_val = evaluate_model(dl_test, model, criterion, inv_label_fn)

        # scheduler.step()

        # Print training and validation results
        print('--------------------------')
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {loss:.4f}, Train MRE: {perc_error:.4f}, Train D2: {d2_score:.4f}')
        print(f'Valid Loss: {valid_loss:.4f}, Valid MRE: {perc_error_val:.4f}, Valid D2: {d2_score_val:.4f}')
        print('--------------------------')


    print(f"Final D2 Validation Score is {d2_score_val}")
    # simple model uncertainty init
    if uncertainty_technique == "nac":
        model = param_sweep_nac(model, dl_train, dl_val)
    elif uncertainty_technique == "mcdropout":
        model = MCWrapper(model)

    else:
        assert isinstance(model, EnsembleWrapper)

    if uncertainty_kind == "epistemic":
        evaluate_uncertainty_epistemic(model, dl_test, dl_ood)
    else:
        evaluate_uncertainty_aleatoric(model, dl_test)

if __name__ == "__main__":
    tyro.cli(main)