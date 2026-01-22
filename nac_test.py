# SPDX-FileCopyrightText: 2026 DLR e.V.
#
# SPDX-License-Identifier: MIT

from ast import mod
import torch
from nac_uncertainty_regression.nac import NACWrapper, NACMode
import pytest

class TestNACWrapper:
    def setup_method(self):
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 2, kernel_size=1, bias=False),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(2, 2, bias=False)
        )
        self.model[0].weight = torch.nn.Parameter(torch.tensor([[[[1.0]]], [[[1.0]]]]))
        self.model[-1].weight = torch.nn.Parameter(torch.tensor([[0.5, 0.5],
                                                                 [1, -1]]))
        self.wrapper = NACWrapper(self.model, ["0", "2"], M=4)

    def test_model_initialization(self):
        assert isinstance(self.wrapper._model, torch.nn.Sequential)

    def test_activation_caching(self):
        model = NACWrapper.ActivationCachingWrapper(self.model, "test")
        x = torch.tensor([[[[0.5, 0.5],
                            [1, 1]]]])
        _ = model(x)
        # check if activations are cached
        wrapped_layer: NACWrapper.ActivationCachingWrapper = getattr(self.wrapper._model, "0")
        torch.testing.assert_close(torch.tensor([[[[0.5, 0.5],
                                                    [1, 1]],
                                                  [[0.5, 0.5],
                                                    [1, 1]]]]), 
                                   wrapped_layer.cache[0])

    def test_saving_histograms(self):
        x = torch.tensor([[[[0.5, 0.5],
                            [1, 1]]]])
        self.wrapper.train()
        _ = self.wrapper(x)

        # check if histogram bins have been updated
        assert self.wrapper.histograms["0"] is not None
        # batch size was one, so each histogram should have been updated with one entry
        assert torch.sum(self.wrapper.histograms["0"][0]) == 1
        assert torch.sum(self.wrapper.histograms["0"][1]) == 1
        
    def test_uncertainty_computation(self):
        x = torch.rand(1, 1, 28, 28)
        self.wrapper.train()
        _ = self.wrapper(x)
        self.wrapper.eval()
        output = self.wrapper(x)
        assert "uncertainty" in output
        assert isinstance(output["uncertainty"], torch.Tensor)

    def test_train_mode(self):
        self.wrapper.eval()
        assert not self.wrapper.training
        self.wrapper.train()
        assert self.wrapper.training

    def test_integration(self):
        # let's train a neural network to predict if the first component of a 2-vector is greater than 5
        # the net should only be one linear lay
        import random
        torch.manual_seed(1)
        random.seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(2, 10), torch.nn.ReLU(), torch.nn.Linear(10, 2))
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        # Generate training data
        # We'll create a batch of samples where:
        # - x[:,0] is a random float
        # - x[:,1] is always 0
        # Target is 1 if x[:,0] < 2, else 0
        def generate_data(batch_size=64):
            x = torch.zeros(batch_size, 2)
            x[:, 0] = torch.randn(batch_size) * 3  # Random values with std dev 3, mean 0 (can be negative or positive)
            y = (x[:, 0] < 2).long()  # Targets: 1 if first component < 2, else 0
            return x, y

        # Training loop
        num_epochs = 1000
        for epoch in range(num_epochs):
            model.train()
            x_train, y_train = generate_data(batch_size=64)
            optimizer.zero_grad()
            outputs = model(x_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 100 == 0:
                with torch.no_grad():
                    preds = torch.argmax(outputs, dim=-1)
                    accuracy = (preds.float() == y_train).float().mean()
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")

        # Test the trained model
        model.eval()
        test_samples = torch.tensor([
            [1.5, 0.0],  # < 2, label 1
            [2.5, 0.0],  # >= 2, label 0
            [0.0, 0.0],  # < 2, label 1
            [3.0, 0.0],  # >= 2, label 0
        ])

        with torch.no_grad():
            test_outputs = model(test_samples)
            test_preds = torch.argmax(test_outputs, dim=-1) 
            print("Test samples predictions:", test_preds.squeeze().tolist())

        model = NACWrapper(model, layer_name_list=["0", "2"], O=100)
        model.train()
        for _ in range(10):
            x_train, y_train = generate_data(batch_size=64)
            _ = model(x_train)

        model.eval()
        unc_samples = torch.tensor([
            [1.5, 0.0],  # < 2, label 1, ID
            [2.5, 0.0],  # >= 2, label 0, ID
            [-5.0, 10.0],  # < 2, label 1, OOD
            [10.0, -15.0],  # >= 2, label 0, OOD
        ])
        results = model(unc_samples)
        unc_array = results["uncertainty"]
        print("Uncertainty array:", unc_array)
        assert unc_array[0] < unc_array[2]
        assert unc_array[1] < unc_array[3]

    def test_integration_single_neuron(self):
        # let's train a neural network to predict if the first component of a 2-vector is greater than 5
        # the net should only be one linear lay
        import random
        torch.manual_seed(3)
        random.seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(2, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1))
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        # Generate training data
        # We'll create a batch of samples where:
        # - x[:,0] is a random float
        # - x[:,1] is always 0
        # Target is 1 if x[:,0] < 2, else 0
        def generate_data(batch_size=64):
            x = torch.zeros(batch_size, 2)
            x[:, 0] = torch.randn(batch_size) * 3  # Random values with std dev 3, mean 0 (can be negative or positive)
            y = (x[:, 0] < 2).float()  # Targets: 1 if first component < 2, else 0
            return x, y

        # Training loop
        num_epochs = 1000
        for epoch in range(num_epochs):
            model.train()
            x_train, y_train = generate_data(batch_size=64)
            optimizer.zero_grad()
            outputs = model(x_train)
            loss = criterion(outputs.squeeze(-1), y_train)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 100 == 0:
                with torch.no_grad():
                    preds = (torch.sigmoid(outputs) > 0.5).float().unsqueeze(-1)
                    accuracy = (preds.float() == y_train).float().mean()
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")

        model = NACWrapper(model, layer_name_list=["0", "2"], O=100)
        model.train()
        for _ in range(10):
            x_train, y_train = generate_data(batch_size=64)
            _ = model(x_train)

        model.eval()
        unc_samples = torch.tensor([
            [3.5, 0.0],  # < 2, label 1, ID
            [3, 0.0],  # >= 2, label 0, ID
            [5000.0, 1000.0],  # < 2, label 1, OOD
            [10000.0, -1500.0],  # >= 2, label 0, OOD
        ])
        results = model(unc_samples)
        unc_array = results["uncertainty"]
        print("Uncertainty array:", unc_array)
        assert unc_array[0] < unc_array[2]
        assert unc_array[1] < unc_array[3]

    def test_integration_regression(self):
        # let's train a neural network to predict if the first component of a 2-vector is greater than 5
        # the net should only be one linear lay
        import random
        torch.manual_seed(42)
        random.seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.LeakyReLU(), torch.nn.Linear(32, 1))
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        # Generate training data
        # We'll create a batch of samples where:
        # - x[:,0] is a random float
        # - x[:,1] is always 0
        # Target is 4 * x[:,0] + x[:,1]
        def generate_data(batch_size=64):
            x = torch.zeros(batch_size, 2)
            x[:, 0] = torch.randn(batch_size) * 3  # Random values with std dev 3, mean 0 (can be negative or positive)
            y = 4 * x[:, 0] + x[:, 1]
            return x, y

        # Training loop
        num_epochs = 1000
        for epoch in range(num_epochs):
            model.train()
            x_train, y_train = generate_data(batch_size=64)
            optimizer.zero_grad()
            outputs = model(x_train)
            loss = criterion(outputs.squeeze(-1), y_train)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}, MSE: {loss.item():.4f}")

        model = NACWrapper(model, layer_name_list=["0", "2"], O=100, mode=NACMode.REGRESSION)
        model.train()
        for _ in range(10):
            x_train, y_train = generate_data(batch_size=64)
            _ = model(x_train)

        model.eval()
        unc_samples = torch.tensor([
            [3.5, 0.0],  # target 7, ID
            [3, 0.0],  # target 6, ID
            [5.0, 1.0],  # target 21, OOD
            [1.0, -1.0],  # target 3, OOD
        ])
        results = model(unc_samples)
        unc_array = results["uncertainty"]
        print("Uncertainty array:", unc_array)
        assert unc_array[0] < unc_array[2]
        assert unc_array[1] < unc_array[3]


# Run the `py.test` command to execute the tests
if __name__ == "__main__":
    pytest.main()