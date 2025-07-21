import torch
from torch.utils.data import DataLoader
from modnet import DeepModNet
from dataset import load_dataset
from trainer import train
from utils import set_seed, plot_loss_curve

def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mod = 7
    input_dim = 3
    hidden_dims = [8, 8]
    output_dim = 2
    epochs = 3000
    batch_size = 2

    dataset = load_dataset(mod)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DeepModNet(input_dim, hidden_dims, output_dim, mod).to(device)

    losses = train(model, dataloader, epochs, device=device)

    plot_loss_curve(losses)

    print("\n最終出力:")
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            x, y_true = dataset[i]
            x = x.to(device)
            out = model(x)
            print(f"Input: {x.tolist()}, True Output: {y_true.tolist()}, Predicted Output: {out.tolist()}")

if __name__ == "__main__":
    main()
