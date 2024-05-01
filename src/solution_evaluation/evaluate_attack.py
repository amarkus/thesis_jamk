# Evaluation script: Attack
# - Prepare demonstration dataset
# - Load data from demonstration dataset
# - Compute evaluation results for CODAIT predictions using original & adversarial images (from test dataset)
# - Compute and save results to file
# - Print summeries & plot results (images and tables)

from solution_evaluation.evaluation import Evaluation


# Compute evaluation results for CODAIT predictions.
# Then compute reconstruction errors using the developed autoencoder model.
evaluation = Evaluation(autoencoder_model_name="model_autoencoder_thesis.keras")
evaluation.codait_result_for_images(count=100, recursive=True)

# Save .csv data of all result sets
evaluation.save_normal_images_as_csv()
evaluation.save_mitosis_images_as_csv()
evaluation.save_all_images_as_csv()
evaluation.save_adversarial_images_as_csv()

# Print to console for debugging & evaluation
evaluation.print_average_prediction_error(
    label="normal", arr=evaluation.normal_images_predictions
)
evaluation.print_average_prediction_error(
    label="mitosis", arr=evaluation.mitosis_images_predictions
)
evaluation.print_average_prediction_error(
    label="both", arr=evaluation.all_images_predictions
)

# Print summaries of attack performance to file
evaluation.print_summary(
    title="CODAIT Prediction change (mitosis-to-normal)     ",
    before_data=evaluation.mitosis_images_predictions,
    after_data=evaluation.adv_mitosis_images_predictions,
    print_to_file=True,
)

evaluation.print_summary(
    title="CODAIT Prediction change (normal-to-mitosis)     ",
    before_data=evaluation.normal_images_predictions,
    after_data=evaluation.adv_normal_images_predictions,
    print_to_file=True,
)

evaluation.print_summary(
    title="Autoencoder reconstruction error (before-after)",
    before_data=evaluation.all_images_reconstruction_error,
    after_data=evaluation.all_adv_images_reconstruction_error,
    print_to_file=True,
)

# Visualize attack before & after results with box plots.
evaluation.boxplot_results(
    evaluation.mitosis_images_predictions,
    evaluation.adv_mitosis_images_predictions,
    label1="initial prediction",
    label2="prediction after attack",
    title="CODAIT Prediction change (mitosis-to-normal)",
)

evaluation.boxplot_results(
    evaluation.normal_images_predictions,
    evaluation.adv_normal_images_predictions,
    label1="initial prediction",
    label2="prediction after attack",
    title="CODAIT Prediction change (normal-to-mitosis)",
)

evaluation.boxplot_results(
    evaluation.all_images_reconstruction_error,
    evaluation.all_adv_images_reconstruction_error,
    label1="Initial reconstruction error",
    label2="Reconstruction error after attack",
    title="Autoencoder reconstruction error (before-after)",
)
