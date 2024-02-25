import torch


class LinWarmupExpDecay():

    def __init__(
        self,
        base_learning_rate,
        global_step=torch.LongTensor([0]),
        warmup_steps=torch.LongTensor([10000]),
        decay_rate=0.5,
        decay_steps=torch.LongTensor([100000])
    ):

        self.base_learning_rate = base_learning_rate
        self._learning_rate = base_learning_rate
        self._global_step = global_step

        self.warmup_steps = warmup_steps
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

        self.update()

    @property
    def learning_rate(self):
        return self._learning_rate.item()

    @property
    def global_step(self):
        return self._global_step

    @global_step.setter
    def global_step(self, value):
        self._global_step = value
        self.update()

    def update(self):
        if self._global_step < self.warmup_steps:
            self._learning_rate = self.base_learning_rate \
                * self._global_step.double() / self.warmup_steps.double()
        else:
            self._learning_rate = self.base_learning_rate * self.decay_rate \
                ** (self._global_step.double() / self.decay_steps.double())

    def step(self):
        self.global_step = self.global_step + 1
        self.update()
