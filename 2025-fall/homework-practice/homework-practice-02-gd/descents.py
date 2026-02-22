import numpy as np
from abc import ABC, abstractmethod

# ===== Learning Rate Schedules =====
class LearningRateSchedule(ABC):
    @abstractmethod
    def get_lr(self, iteration: int) -> float:
        pass


class ConstantLR(LearningRateSchedule):
    def __init__(self, lr: float):
        self.lr = lr

    def get_lr(self, iteration: int) -> float:
        return self.lr


class TimeDecayLR(LearningRateSchedule):
    def __init__(self, lambda_: float = 1.0):
        self.s0 = 1
        self.p = 0.5
        self.lambda_ = lambda_

    def get_lr(self, iteration: int) -> float:
        new_lr = self.lambda_*(self.s0/(self.s0+iteration))**p
        return new_lr

# ===== Base Optimizer =====
class BaseDescent(ABC):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR):
        self.lr_schedule = lr_schedule()
        self.iteration = 0
        self.model = None

    def set_model(self, model):
        self.model = model

    @abstractmethod
    def update_weights(self):
        pass

    def step(self):
        self.update_weights()
        self.iteration += 1


# ===== Specific Optimizers =====
class VanillaGradientDescent(BaseDescent):
    def update_weights(self):
        # TODO: реализовать vanilla градиентный спуск
        # Можно использовать атрибуты класса self.model
        X_train = self.model.X_train
        y_train = self.model.y_train
        gradient = self.model.compute_gradients(X_train, y_train)
        lr = self.lr_schedule.get_lr(self.iteration)
        self.model.w -= lr*gradient 
        return -lr*gradient


class StochasticGradientDescent(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR, batch_size=1):
        super().__init__(lr_schedule)
        self.batch_size = batch_size

    def update_weights(self):
        # TODO: реализовать стохастический градиентный спуск
        # 1) выбрать случайный батч
        # 2) вычислить градиенты на батче
        # 3) обновить веса модели
        X_train = self.model.X_train
        y_train = self.model.y_train
        idxs = np.random.choice(len(y_train)-1, self.batch_size, replace=False)
        lr = self.lr_schedule.get_lr(self.iteration)
        gradient = self.model.compute_gradients(X_train[idxs], y_train[idxs])
        self.model.w -= lr*gradient
        return -lr*gradient


class SAGDescent(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR, batch_size=1):
        super().__init__(lr_schedule)
        self.grad_memory = None
        self.grad_sum = None
        self.batch_size = batch_size

    def update_weights(self):
        # TODO: реализовать SAG
        X_train = self.model.X_train
        y_train = self.model.y_train
        num_objects, num_features = X_train.shape
        if self.grad_memory is None:
            # TODO: инициализировать хранилища при первом вызове
            self.grad_memory = []
            for i in range(num_objects):
                self.grad_memory.append(self.model.compute_gradients(X_train[i:i+1], y_train[i:i+1]))
            self.grad_memory = np.array(self.grad_memory)
            self.grad_sum = np.sum(self.grad_memory)
        idxs = np.random.choice(num_objects-1, self.batch_size, replace=False)
        gradient = self.model.compute_gradients(X_train[idxs], y_train[idxs])
        self.grad_sum+=gradient*self.batch_size-np.sum(self.grad_memory[idxs])
        lr = self.lr_schedule.get_lr(self.iteration)
        self.model.w -= lr*self.grad_sum/num_objects
        return -lr*self.grad_sum/num_objects
        # TODO: реализовать SAG

class MomentumDescent(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR, beta=0.9):
        super().__init__(lr_schedule)
        self.beta = beta
        self.velocity = None

    def update_weights(self):
        # TODO: реализовать градиентный спуск с моментумом
        X_train = self.model.X_train
        y_train = self.model.y_train
        gradient = self.model.compute_gradients(X_train, y_train)
        lr = self.lr_schedule.get_lr(self.iteration)
        if self.velocity is None:
            self.velosity = np.zeros(X_train.shape[1])
        self.velocity = beta*self.velocity + lr*gradient
        self.model.w-=self.velocity
        return -self.velocity


class Adam(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr_schedule)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def update_weights(self):
        X_train = self.model.X_train
        y_train = self.model.y_train
        gradient = self.model.compute_gradients(X_train, y_train)
        lr = self.lr_schedule.get_lr(self.iteration)
        if self.m is None:
            self.m = np.zeros(X_train.shape[1])
            self.v = np.zeros(X_train.shape[1])
        self.m = self.beta1*self.m + (1-self.beta1)*gradient
        self.v = self.beta2*self.v + (1-self.beta2)*gradient**2
        mk = self.m/(1-self.beta1**self.iteration)
        vk = self.v/(1-self.beta2**self.iteration)
        h = lr/(np.sqrt(vk)+self.eps)
        self.model.w -= h*mk
        return -h*mk
