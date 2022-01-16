import torch_geometric
from datasets import InMemoryVesselDataset
import torch
from utils import training, visualisation, log
import tqdm
import os
from utils.metrics import Metrics


class Experiment:
    def __init__(self, model, dataset, batch_size, tag, transforms=None, lr=0.001, epochs=1000):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.tag = tag

        # Data transforms
        transforms_list = []  # dummy list
        if transforms is not None:
            transforms_list.extend(transforms)
        self.transform = torch_geometric.transforms.Compose(transforms_list)

        # Optional parameters
        self.lr = lr
        self.epochs = epochs

    def run(self, device):

        if self.dataset == 'bifurcating_arteries':
            self.fit_bifurcating(device)

        elif self.dataset == 'single_arteries':
            self.fit_single(device)

        else:
            raise RuntimeError("Dataset not found.")

    def fit_bifurcating(self, device):

        # Dataset IDs
        path = ...
        pattern = "visuals/sample_*.vtp"  # in the "raw" folder

        # Training, validation and test split (total 2170 samples)
        train_split = [0, 1600]
        valid_split = [1600, 1800]
        test_split = [1800, 2000]

        # Data loader parameters
        params = {'batch_size': self.batch_size, 'shuffle': False,
                  'num_workers': 0, 'pin_memory': False}

        args = [self.tag, path, pattern, train_split, valid_split, test_split, params]
        self.fit(device, args)

    def fit_single(self, device):

        # Dataset IDs
        # path = ...
        # pattern = "visuals/sample_*.vtp"  # in the "raw" folder
        path = "/Users/arpj/code/princeton/coronary-mesh-convolution/datasets"
        pattern = "visuals/sample_*.vtp"  # in the "raw" folder


        # Training, validation and test split (total 2000 samples)
        train_split = [0, 1600]
        valid_split = [1600, 1800]
        test_split = [1800, 2000]

        # Data loader parameters
        params = {'batch_size': self.batch_size, 'shuffle': False,
                  'num_workers': 0, 'pin_memory': False}

        args = [self.tag, path, pattern, train_split, valid_split, test_split, params]
        self.fit(device, args)

    def fit(self, device, args):
        tag, path, pattern, train_split, valid_split, test_split, params = args

        # Create datasets
        train = InMemoryVesselDataset(path, pattern, train_split, "train", pre_transform=self.transform)
        valid = InMemoryVesselDataset(path, pattern, valid_split, "valid", pre_transform=self.transform)
        test = InMemoryVesselDataset(path, pattern, test_split, "test", pre_transform=self.transform)
        print("-----datasets created-----")

        # Data loaders
        train_loader = torch_geometric.data.DataLoader(train, **params)
        valid_loader = torch_geometric.data.DataLoader(valid, **params)
        test_loader = torch_geometric.data.DataLoader(test, batch_size=1)

        # Network model (FeaSt convolutional network)
        model = self.model.to(device)
        print(f"-----model to device {device}-----")

        # Optimisation settings
        objective = torch.nn.MSELoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=self.lr)

        # Training
        print(f"Training started. Epochs to run: {self.epochs}")
        training.fit(model, [train_loader, valid_loader],
                     objective, self.epochs, optimiser, device,
                     tag=tag)
        print("Training ended!")

        # Write predictions to VTP files for visualisation
        print("Writing predictions to VTP files for viz")
        model.load_state_dict(torch.load("data/" + tag + ".pt", map_location='cpu'))
        model.eval()  # set to evaluation mode
        if not os.path.exists('vis'):
            os.makedirs('vis')
        i = 0
        for sample in tqdm.tqdm(test_loader):
            prediction = model(sample.to(device))
            fields = visualisation.default_fields(sample, prediction)
            # fields['pooling'] = visualisation.pooling_scales(sample)
            filename = "vis/prediction" + str(i) + ".vtp"
            visualisation.new_file(sample.pos, sample.face, filename, fields)
            i += 1

        # Tabulate the evaluation metrics
        print("Tabulating metrics")
        evaluation = Metrics([test_loader]).statistics(model, device)
        print(evaluation)

        # Log the experiment for identification
        print("Logging experiment")
        log.experiment(self.model, self.dataset, self.batch_size, self.transform, self.epochs, self.lr, optimiser,
                       objective)
        print("Ciao!")
