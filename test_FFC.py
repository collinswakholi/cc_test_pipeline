import cv2
from FFC.FF_correction import FlatFieldCorrection

# ----------------- Example usage (if run as __main__) -----------------
if __name__ == "__main__":
    # Example: Load a white-background image and perform FFC
    path_ = r"Data/Backgrounds/White/Blank.JPG"
    w_img = cv2.imread(path_, cv2.IMREAD_COLOR)

    ffc_params = {
        "model_path": "best_models/PD_trained_512_dauntless-sweep-1/weights/best.pt",
        "manual_crop": False,
        "smooth_window": 11,
        "bins": 50,
        "show": True,
    }

    fit_params = {
        "degree": 5,
        "interactions": True,
        "fit_method": "nn",  # linear, nn, pls, svm
        "max_iter": 1000,
        "tol": 1e-8,
        "verbose": False, 
        "rand_seed": 0,
    }

    ffc = FlatFieldCorrection(img=w_img, **ffc_params)
    multiplier = ffc.compute_multiplier(**fit_params)
    # np.save("mult.npy", multiplier)

    corrected_img = ffc.apply_ffc(img=w_img, multiplier=multiplier, show=True)