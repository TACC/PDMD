import torch

def train(model, args, train_loader, optimizer, epoch=None):
    model_name = model.model_name
    assert model_name in ["ChemGNN_energy", "ChemGNN_force"]
    model.train()
    total_loss = 0
    gradients_list = []
    if model_name == "ChemGNN_energy":
        for batch_i, data in enumerate(train_loader):
            data = data.to(args.device)
            optimizer.zero_grad()
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
            loss.backward()
            total_loss += loss.item() * data.num_graphs

            for name, parameter in model.named_parameters():
                if parameter.requires_grad and name == 'energy_predictor.4.weight':
                    gradients = torch.norm(parameter.grad, p=2)
                    gradients = gradients.item()
                    gradients_list.append(gradients)
                    break
            optimizer.step()
        loss = total_loss / len(train_loader.dataset)
    if model_name == "ChemGNN_force":
        for batch_i, data in enumerate(train_loader):
            data = data.to(args.device)
            optimizer.zero_grad()
            input_dict = dict({
                "x": data.x,
                "edge_index": data.edge_index,
                "edge_attr": data.edge_attr,
                "batch": data.batch
            })
            out = model(input_dict)

            loss = (out.squeeze() - data.z).abs().mean()
            loss.backward()
            total_loss += loss.item() * data.num_nodes

            for name, parameter in model.named_parameters():
                if parameter.requires_grad and name == 'force_predictor.2.weight':
                    gradients = torch.norm(parameter.grad, p=2)
                    gradients = gradients.item()
                    gradients_list.append(gradients)
                    break
            optimizer.step()
        loss = total_loss / sum([data.num_nodes for data in train_loader.dataset])

    return model, loss, sum(gradients_list) / len(gradients_list)

