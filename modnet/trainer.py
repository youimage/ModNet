import torch

def loss_fn(y_pred, y_true, mod):
    return torch.sum((y_pred - y_true) % mod)

def train(model, dataloader, epochs=1000, device=None):
    losses = []
    mod = model.mod

    for epoch in range(epochs):
        epoch_total_loss = 0
        with torch.no_grad():
            dWs = []
            for layer in model.layers:
                dW = torch.randint(-1, 2, layer.W.shape, dtype=torch.int64, device=device)
                dWs.append(dW)

            W_try = []
            for layer, dW in zip(model.layers, dWs):
                W_try.append((layer.W + dW) % mod)

            for X_batch, Y_batch in dataloader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)

                # バッチ内損失計算
                batch_loss = 0
                for x, y in zip(X_batch, Y_batch):
                    h = x
                    for W in W_try[:-1]:
                        h = (h @ W) % mod
                    out = (h @ W_try[-1]) % mod
                    batch_loss += loss_fn(out, y, mod)

                epoch_total_loss += batch_loss.item()

            # 現モデルの損失計算
            current_loss = 0
            for X_batch, Y_batch in dataloader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                for x, y in zip(X_batch, Y_batch):
                    h = x
                    for layer in model.layers[:-1]:
                        h = (h @ layer.W) % mod
                    out = (h @ model.layers[-1].W) % mod
                    current_loss += loss_fn(out, y, mod).item()

            if epoch_total_loss < current_loss:
                for layer, new_W in zip(model.layers, W_try):
                    layer.W.data = new_W

            losses.append(epoch_total_loss)

            if epoch == 0 or epoch % 100 == 0 or epoch_total_loss == 0:
                print(f"Epoch {epoch}: Loss = {epoch_total_loss}")

            if epoch_total_loss == 0:
                break

    return losses
