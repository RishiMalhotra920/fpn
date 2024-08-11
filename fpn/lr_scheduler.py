import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    _LRScheduler,
)

# Define the custom learning rate schedule


class CustomExponentialLR(_LRScheduler):
    """
    It increases the learning rate exponentially from 0.001 to 0.01 in 10 epochs
    and then keeps it constant at 0.01 for the remaining epochs.
    """

    def __init__(
        self,
        optimizer,
        init_lr,
        final_growth_lr,
        growth_epochs,
        last_epoch=-1,
    ):
        self.init_lr = init_lr
        self.final_lr = final_growth_lr
        self.growth_epochs = growth_epochs

        self.growth_factor = (final_growth_lr / init_lr) ** (1 / growth_epochs)

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # self.base_lrs stores the learning rate for each group in the optimizer.
        # self.last_epoch is the last epoch. it is updated by the base class.
        print(self.last_epoch)
        if self.last_epoch < self.growth_epochs:
            return [base_lr * (self.growth_factor**self.last_epoch) for base_lr in self.base_lrs]
        elif self.last_epoch < self.growth_epochs + 70:
            return [0.001 for base_lr in self.base_lrs]
        elif self.last_epoch < self.growth_epochs + 100:
            return [0.0001 for base_lr in self.base_lrs]
        elif self.last_epoch <= self.growth_epochs + 130:
            return [0.00001 for base_lr in self.base_lrs]

        return [self.final_lr for base_lr in self.base_lrs]


def get_custom_lr_scheduler(optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    """
    It increases the learning rate exponentially from 0.001 to 0.01 in 10 epochs
    and then keeps it constant at 0.01 for the remaining epochs.
    """
    scheduler = CustomExponentialLR(
        optimizer,
        init_lr=0.0001,
        final_growth_lr=0.001,
        growth_epochs=5,
    )
    return scheduler


def get_fixed_lr_scheduler(optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1)


if __name__ == "__main__":
    # Test the learning rate schedule above
    model = torch.nn.Linear(10, 2)
    initial_lr = 0.0001
    optimizer = optim.SGD(model.parameters(), lr=initial_lr)
    # scheduler = get_fixed_lr_scheduler(optimizer)
    scheduler = get_custom_lr_scheduler(optimizer)

    # Training loop
    learning_rates = []
    for epoch in range(135):  # example for 20 epochs
        # Training code here
        optimizer.zero_grad()
        # Assume some fake loss
        loss = model(torch.randn(1, 10))
        # loss.backward()
        optimizer.step()

        # Update the learning rate
        scheduler.step()

        # Print learning rate
        learning_rates.append(scheduler.get_last_lr()[0])

    plt.figure(figsize=(20, 10))
    plt.plot(learning_rates, marker="o")  # 'o' marker to show points

    # Setting x-ticks to only integers
    # Assuming the epochs are the indices of the list
    plt.xticks(range(len(learning_rates)))

    # Annotating y-values on the line
    for i, lr in enumerate(learning_rates):
        plt.annotate(
            f"{lr:.5f}",  # Formatting to 3 decimal places
            (i, lr),
            textcoords="offset points",  # Positioning text
            xytext=(0, 10),  # Distance from text to points (x,y)
            ha="center",
        )  # Horizontal alignment can be left, right or center

    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Custom Exponential Learning Rate Scheduler")
    plt.grid(True)  # Optional: adds a grid
    plt.show()
