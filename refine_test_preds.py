import os
import cv2
import shutil
from mmseg.apis import init_model, inference_model, show_result_pyplot
import mmcv


def zoom_into_foreground(prediction_image_path, original_image_path):
    prediction_image = cv2.imread(prediction_image_path)

    # find the contours (boundaries) of objects in a binary image
    contours, _ = cv2.findContours(cv2.cvtColor(prediction_image, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    original_image = cv2.imread(original_image_path)

    foreground_objects = []

    # iterate over each contour and create cropped and resized images
    for contour in contours:
        # takes a single contour (a list of points) as input and returns four values: x, y, w, and h.
        x, y, w, h = cv2.boundingRect(contour)

        foreground_object = original_image[y:y+h, x:x+w]

        # resize the foreground object to match the prediction image size
        foreground_zoomed = cv2.resize(foreground_object, (prediction_image.shape[1], prediction_image.shape[0]))

        foreground_objects.append(foreground_zoomed)

    return foreground_objects


def replace_foreground_objects(prediction_image_path, refine_prediction_path, new_predictions):
    prediction_image = cv2.imread(prediction_image_path).copy()

    # find the contours of the foreground objects
    contours, _ = cv2.findContours(cv2.cvtColor(prediction_image, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # iterate over each contour and replace with new predictions
    for i, contour in enumerate(contours):
        # calculate the bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)

        # get the corresponding new prediction for the current contour
        new_prediction = new_predictions[i]

        # resize the new prediction to match the size of the contour region
        new_prediction_resized = cv2.resize(new_prediction, (w, h))

        # replace the contour region with the new prediction
        prediction_image[y:y+h, x:x+w] = new_prediction_resized

    # save the modified prediction image
    cv2.imwrite(refine_prediction_path, prediction_image)
    
    
def refine_and_save_predictions(models, base_prediction_dir, dataset_names, model_names):    
    model_idx = 0  # Initialize model index to keep track of the current model
    for model_name in model_names:
        for dataset_name in dataset_names:
            # Define paths to the original prediction and where to save refined predictions
            prediction_dir = os.path.join(base_prediction_dir, model_name, dataset_name)
            refined_prediction_dir = os.path.join(prediction_dir, "refined")
            if not os.path.exists(refined_prediction_dir):
                os.makedirs(refined_prediction_dir)
            
            prediction_files = [f for f in os.listdir(prediction_dir) if f.endswith('.png') and not f.startswith('refined')]

            for idx, prediction_file in enumerate(prediction_files):
                prediction_image_path = os.path.join(prediction_dir, prediction_file)
                original_image_path = os.path.join('/pv-segmentation/datasets', dataset_name, 'img', prediction_file)
                
                foreground_objects = zoom_into_foreground(prediction_image_path, original_image_path)
                new_predictions = []
                for foreground_object in foreground_objects:
                    result = inference_model(model=models[model_idx], img=foreground_object) 
                    vis_result = show_result_pyplot(model=models[model_idx], img=foreground_object, result=result, wait_time=1e-10, opacity=1.0, with_labels=False)
                    new_predictions.append(mmcv.bgr2rgb(vis_result))
                
                # Correctly constructing the refine_prediction_path
                refine_prediction_path = os.path.join(refined_prediction_dir, prediction_file)
                
                
                # Replace the foreground objects in the prediction image with new predictions
                replace_foreground_objects(prediction_image_path, refine_prediction_path, new_predictions)
                
            model_idx += 1
                
if __name__ == "__main__":
    base_prediction_dir = '/path/to/folder/with/prediction_images' # Update accordingly
    dataset_names = ['PV01', 'PV03', 'PV08', 'google', 'ign']
    model_names = ['unet', 'deeplabv3+', 'mask2former']
    
    
    # Paths to best checkpoints for each model x dataset
    unet_checkpoints = {
        'pv01': '',
        'pv03': '',
        'pv08': '',
        'google': '',
        'ign': ''
    }
    
    deeplabv3plus_checkpoints = {
        'pv01': '',
        'pv03': '',
        'pv08': '',
        'google': '',
        'ign': '',
    }
    
    mask2former_checkpoints = {
        'pv01': '',
        'pv03': '',
        'pv08': '',
        'google': '',
        'ign': ''
    }
    
    # These configutations are not the same we used to train the model, these will be under work_dirs, also specific to model and dataset
    model_specs = {
        'unet': {
            'checkpoints': unet_checkpoints,
            'configs': {
                'pv01': '',
                'pv03': '',
                'pv08': '',
                'google': '',
                'ign': '',
            }
        },
        'deeplabv3+': {
            'checkpoints': deeplabv3plus_checkpoints,
            'configs': {
                'pv01': '',
                'pv03': '',
                'pv08': '',
                'google': '',
                'ign': '',
            }
        },
        'mask2former': {
            'checkpoints': mask2former_checkpoints,
            'configs': {
                'pv01': '',
                'pv03': '',
                'pv08': '',
                'google': '',
                'ign': '',
            },
        },
    }
    
    models = []

    for model_name, specs in model_specs.items():
        # Iterate through each dataset within a model
        for dataset_name, checkpoint_path in specs['checkpoints'].items():
            # Select the specific config for this dataset
            config_path = specs['configs'][dataset_name]  # Updated to select the correct config

            # Determine the CUDA device based on the model count
            model = init_model(config_path, checkpoint_path, 'cuda:0')
            models.append(model)
    
    refine_and_save_predictions(models, base_prediction_dir, dataset_names, model_names)


