import torch

@torch.no_grad()
def val(model, config, loader, epoch=None):
    model_name = model.model_name
    assert model_name in ["ChemGNN_energy", "ChemGNN_forces"]
    model.eval()
    total_error = 0
    if model_name == "ChemGNN_energy":
        for data in loader:
            data = data.to(config.device)
            input_dict = dict({
                "x": data.x,
                "edge_index": data.edge_index,
                "edge_attr": data.edge_attr,
                "batch": data.batch
            })
            out = model(input_dict)
            batch = input_dict["batch"]
            node_counts = torch.bincount(batch)

            loss = (((out.squeeze() - data.y) / node_counts).abs()).mean()
            total_error += loss.item() * data.num_graphs
        loss = total_error / len(loader.dataset)
    if model_name == "ChemGNN_forces":
        for data in loader:
            data = data.to(config.device)
            input_dict = dict({
                "x": data.x,
                "edge_index": data.edge_index,
                "edge_attr": data.edge_attr,
                "batch": data.batch
            })
            out = model(input_dict)
            total_error += (out.squeeze() - data.z).abs().mean().item() * data.num_nodes
        loss = total_error / sum([data.num_nodes for data in loader.dataset])
    return loss
