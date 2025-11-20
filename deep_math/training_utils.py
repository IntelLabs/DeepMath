"""Introducing temperature scheduling."""

from trl import GRPOTrainer


class TemperatureScheduler:
    """
    Linear temperature scheduler that linearly interpolates temperature
    from T_A at step A to T_B at step B.
    """

    def __init__(self, step_a, temp_a, step_b, temp_b):
        """
        Initialize the linear temperature scheduler.

        Args:
            step_a (int): Starting step
            temp_a (float): Temperature at step A
            step_b (int): Ending step
            temp_b (float): Temperature at step B
        """
        if step_a >= step_b:
            raise ValueError("step_a must be less than step_b")

        self.step_a = step_a
        self.temp_a = temp_a
        self.step_b = step_b
        self.temp_b = temp_b

    def get_temperature(self, current_step):
        """
        Get the current temperature for the given step.

        Returns:
            float: Current temperature value
        """
        if current_step <= self.step_a:
            return self.temp_a
        elif current_step >= self.step_b:
            return self.temp_b
        else:
            # Linear interpolation
            progress = (current_step - self.step_a) / (self.step_b - self.step_a)
            return self.temp_a + progress * (self.temp_b - self.temp_a)

    def __str__(self):
        """Human-readable string representation."""
        return (
            f"TemperatureScheduler(step_a={self.step_a}, temp_a={self.temp_a}, "
            f"step_b={self.step_b}, temp_b={self.temp_b}"
        )


class GRPOTrainerTemperature(GRPOTrainer):
    """Trainer with temperature scheduler."""

    def __init__(self, temperature_scheduler: TemperatureScheduler = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature_scheduler = temperature_scheduler

    def training_step(self, model, inputs, num_items_in_batch):
        """Override to step temperature scheduler and apply temperature."""

        # Step the temperature scheduler
        if self.temperature_scheduler is not None:
            self.temperature = self.temperature_scheduler.get_temperature(self.state.global_step)

        # Call parent training_step
        loss = super().training_step(model, inputs, num_items_in_batch)

        # Log temperature
        self._metrics["train"]["temperature"].append(self.temperature)

        return loss


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def visualize_temperature_schedule():
        """Visualize the temperature schedule over time."""
        scheduler = TemperatureScheduler(step_a=0, temp_a=1.2, step_b=400, temp_b=0.6)

        steps = []
        temperatures = []

        for i in range(1000):
            temp = scheduler.get_temperature(i)
            steps.append(i + 1)
            temperatures.append(temp)

        plt.figure(figsize=(10, 6))
        plt.plot(steps, temperatures, linewidth=2)
        plt.xlabel("Training Step")
        plt.ylabel("Temperature")
        plt.title("Linear Temperature Scheduler")
        plt.grid(True, alpha=0.3)
        plt.show()

    # Uncomment to run visualization
    visualize_temperature_schedule()
