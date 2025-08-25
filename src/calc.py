import logging
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from tqdm import tqdm

import file_io


class TQDMHandler(logging.Handler):
    def emit(self, record):
        tqdm.write(self.format(record))


LOGGER = logging.getLogger("logger")
if not LOGGER.hasHandlers():
    TQDM_HANDLER = TQDMHandler()
    TQDM_HANDLER.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    )
    LOGGER.addHandler(TQDM_HANDLER)


def sigmoid(params, x):
    L, x0, k, b = params
    return L / (1 + np.exp(-k * (x - x0))) + b


def residuals(params, x, y):
    return sigmoid(params, x) - y


def sigmoid_derivative(x, L, x0, k, b):
    exp_term = np.exp(-k * (x - x0))
    return L * k * exp_term / ((1 + exp_term) ** 2)


def sigmoid_second_derivative(x, L, x0, k, b):
    exp_term = np.exp(-k * (x - x0))
    numerator = exp_term * (exp_term - 1)
    denominator = (1 + exp_term) ** 3
    return L * k**2 * numerator / denominator


def calc():
    io = file_io.FileIO()
    tolerance = 0.5

    pbar = tqdm(desc="Файлы", unit="", leave=True)

    for file_units in io.read_data():
        pbar.set_postfix_str(file_units.file_name)
        pbar.update(1)
        x_fit = np.linspace(min(file_units.cycles), max(file_units.cycles), 500)
        for column in tqdm(file_units, desc="Столбцы", leave=False):
            y_data = file_units[column]
            min_y, max_y = min(y_data), max(y_data)
            p0 = [max_y - min_y, np.median(file_units.cycles), 1, min_y]

            y_data_norm = [(y - min_y) / (max_y - min_y) for y in y_data]
            p0_norm = [1, np.median(file_units.cycles), 1, 0]

            fig, axs = plt.subplots(3, 1)
            message = None

            result_norm = least_squares(
                residuals,
                p0_norm,
                args=(file_units.cycles, y_data_norm),
                max_nfev=20000,
            )
            if not result_norm.success or result_norm.cost > tolerance:
                LOGGER.warning(
                    f"{file_units.file_name}.{column}: optimal parameters not found, norm cost > tolerance ({result_norm.cost:.2f} > {tolerance})"
                )
                message = "optimal parameters not found"

            result = least_squares(
                residuals, p0, args=(file_units.cycles, y_data), max_nfev=20000
            )
            popt = result.x
            L, x0, k, b = popt

            y_sigmoid = sigmoid(popt, x_fit)
            y_derivative = sigmoid_derivative(x_fit, *popt)
            y_second_derivative = sigmoid_second_derivative(x_fit, *popt)

            axs[0].scatter(file_units.cycles, y_data, label="Данные")
            axs[0].plot(x_fit, y_sigmoid, color="red", linewidth=1, label="Сигмоида")
            axs[0].set_title("Сигмоида")

            max_der_x = x0
            max_der_y = (L * k) / 4

            axs[1].scatter([max_der_x], [max_der_y], label="Экстремум")
            axs[1].plot(
                x_fit,
                y_derivative,
                color="red",
                linewidth=1,
                label="Первая производная",
            )
            axs[1].set_title("Первая производная")

            if max_der_y < sigmoid_derivative(x0 - 1, *popt):
                max_der_x, max_der_y = None, None
                LOGGER.warning(
                    f"{file_units.file_name}.{column}: first derivative has no maximum"
                )
                if message is None:
                    message = "first derivative has no maximum"

            if max_der_x is not None:
                min_sec_der_x = x0 - math.log(2 - math.sqrt(3)) / k
            else:
                min_sec_der_x = x0 - math.log(2 + math.sqrt(3)) / k
            min_sec_der_y = sigmoid_second_derivative(min_sec_der_x, *popt)

            axs[2].scatter([min_sec_der_x], [min_sec_der_y], label="Минимум")
            axs[2].plot(
                x_fit,
                y_second_derivative,
                color="red",
                linewidth=1,
                label="Вторая производная",
            )
            axs[2].set_title("Вторая производная")

            for ax in axs:
                ax.grid(True)
            fig.tight_layout()
            file_units.set_result(
                key=column,
                sigm_data=file_io.SigmData(
                    L=L,
                    x0=x0,
                    k=k,
                    b=b,
                    max_der_x=max_der_x,
                    max_der_y=max_der_y,
                    min_sec_der_x=min_sec_der_x,
                    min_sec_der_y=min_sec_der_y,
                    message=message,
                ),
                fig=fig,
            )

        io.save_data(file_units=file_units)
        plt.close("all")

    pbar.close()


if __name__ == "__main__":
    calc()
