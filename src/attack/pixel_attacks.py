import numpy as np
import time
from PIL import Image
from scipy import optimize
from src.lib.httphelper import HttpHelper


class FewPixelAttacker:
    def __init__(self, dimensions=(64, 64), verbose=False):
        self.dimensions = dimensions
        self.verbose_logging = verbose  # For debugging
        self.numberOfHttpRequests = 0  # To count how many request were required
        self.start = time.time()  # Calculate the start time
        self.timeToCompleteInSeconds = 0  # Time taken to solution (or halt)

    # perturb_image function from:
    # https://github.com/Hyperparticle/one-pixel-attack-keras/blob/master/helper.py
    def perturb_image(self, xs, img):
        if xs.ndim < 2:
            xs = np.array([xs])

        tile = [len(xs)] + [1] * (xs.ndim + 1)
        imgs = np.tile(img, tile)
        xs = xs.astype(int)

        for x, img in zip(xs, imgs):
            pixels = np.split(x, len(x) // 5)
            for pixel in pixels:
                x_pos, y_pos, *rgb = pixel
                img[x_pos, y_pos] = rgb

        return imgs

    # Create perturbed/attack image and send to CODAIT for analysis
    def call_codait_predict(self, xs, img):
        img_perturbed = self.perturb_image(xs, img)[0]
        img_rgb = Image.fromarray(img_perturbed).convert("RGB")
        http_helper = HttpHelper()
        http_result = http_helper.post_inmemory_image(img_rgb)
        self.numberOfHttpRequests += 1
        prediction = float(http_result)

        return [prediction, img_perturbed]

    def predict_mitosis_probability(self, xs, img, minimize=True):
        prediction = self.call_codait_predict(xs, img)[0]
        if self.verbose_logging:
            print("prediction", "%.10f" % prediction)

        return prediction if minimize else 1 - prediction

    # Check if attack was successful comparing prediction to success treshold
    def attack_success(self, x, img, success_threshold, original_prediction):
        prediction = self.call_codait_predict(x, img)[0]

        if success_threshold(original_prediction, prediction):
            print("attack was successful!")
            return True

    # For debugging and documentation:
    def log_parameters(self, pixel_count, maxiter, popsize, popmul, bounds):
        print("bounds:", bounds)
        print("bounds len:", len(bounds))
        print("pixel_count:", pixel_count)
        print("maxiter:", maxiter)
        print("popsize:", popsize)
        print("popmul:", popmul)

    # Define the attack method and it's parameters
    def attack(
        self,
        img,
        pixel_count=1,
        maxiter=75,
        popsize=400,
        threshold_fn=None,
        threshold_val=None,
        minimize=True,
        color_bounds=[(0, 256), (0, 256), (0, 256)],
    ):
        dim_x, dim_y = self.dimensions
        bounds = [
            (0, dim_x),
            (0, dim_y),
            color_bounds[0],
            color_bounds[1],
            color_bounds[2],
        ] * pixel_count
        popmul = max(1, popsize // len(bounds))

        if self.verbose_logging:
            self.log_parameters(pixel_count, maxiter, popsize, popmul, bounds)

        # Predict & callback functions for the DE algorithm:
        # The objective function to be minimized.
        def prediction_fn(xs):
            return self.predict_mitosis_probability(xs, img, minimize)

        # success_threshold
        def default_treshold_fn(prediction):
            return float(prediction) / 2

        # Convert the image and send to REST API to get 'normal' prediction before any perturbations
        img_rgb = Image.fromarray(img).convert("RGB")
        http_helper = HttpHelper()
        original_prediction = http_helper.post_inmemory_image(img_rgb)
        if self.verbose_logging:
            print("confidence before attack:", "%.10f" % float(original_prediction))

        # Define success threshold function to use (default or parameter)
        if threshold_fn:
            success_threshold = threshold_fn
        else:
            success_threshold = default_treshold_fn(original_prediction)

        # Print expected threshold value
        print(
            "Expected success threshold:",
            "%.10f" % threshold_val(original_prediction),
        )

        # A function to follow the progress of the minimization. xk is the best solution found so far.
        # val represents the fractional value of the population convergence. When val is greater than one the function halts.
        # If callback returns True, then the minimization is halted (any polishing is still carried out).
        def callback_fn(x, convergence):
            if self.verbose_logging:
                print("convergence:", convergence)
            return self.attack_success(x, img, success_threshold, original_prediction)

        # Call Differential Evolution (scipy.optimize.differential_evolution)
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
        #
        # [Defalt parameters of differential_evolution SciPy (version 1.8.1)]
        # scipy.optimize.differential_evolution(func, bounds, args=(), strategy='best1bin', maxiter=1000,
        # popsize=15, tol=0.01, mutation=(0.5, 1), recombination=0.7, seed=None, callback=None,
        # disp=False, polish=True, init='latinhypercube', atol=0, updating='immediate', workers=1, constraints=(), x0=None)

        attack_result = optimize.differential_evolution(
            prediction_fn,
            bounds,
            strategy="best1bin",
            maxiter=maxiter,
            popsize=popmul,
            mutation=(0.3, 1),
            recombination=0.7,
            callback=callback_fn,
            atol=-1,
            disp=self.verbose_logging,
            polish=True,
            init="latinhypercube",
        )

        # Get the last image resulted from differential_evolution and it's probability in CODAIT
        (mitosis_probability, attack_image) = self.call_codait_predict(
            attack_result.x, img
        )

        # Calculate the end time and time taken
        end = time.time()
        self.timeToCompleteInSeconds = end - self.start

        # Return results of differential_evolution based attack for stats
        return [
            attack_image,
            pixel_count,
            attack_result.x,
            mitosis_probability,
            self.numberOfHttpRequests,
            self.timeToCompleteInSeconds,
            original_prediction,
        ]
