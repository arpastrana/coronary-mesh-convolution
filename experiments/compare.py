import torch_geometric
from models import CompareFeaSt, CompareSAGE
from transforms import InletGeodesics, RemoveFlowExtensions, HeatSamplingCluster, FeatureDescriptors
from .template import Experiment


def fit(device):
    # Neural network
    model = CompareFeaSt()

    # Training data
    dataset = 'single_arteries'
    batch_size = 4

    # Experiment tag
    tag = "compare_" + dataset

    # Precomputed graph transforms
    transforms = [torch_geometric.transforms.GenerateMeshNormals(),
                  InletGeodesics(),  # remove
                  RemoveFlowExtensions(),
                  HeatSamplingCluster([1., 0.3, 0.1], [0.04, 0.08, 0.2], loop=True),  # change it
                  FeatureDescriptors(r=0.2)]  # change radius to include the first hop neighbourhood,

    # Neural network training
    experiment = Experiment(model=model,
                            dataset=dataset,
                            batch_size=batch_size,
                            tag=tag,
                            transforms=transforms,
                            epochs=5)  # 400
    experiment.run(device)
