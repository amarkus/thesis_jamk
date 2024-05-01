# Demonstration script: Attack

# [Attack single images]
#   - Run attack against 1 mitosis & 1 normal image (from validation dataset)
#   - Plot results (1+1 images, table)

# [Attack whole dataset]
#   - Prepare demonstration dataset
#   - Load data from demonstration dataset
#   - Compute evaluation results for CODAIT predictions using original & adversarial images (from test dataset)
#   - Compute results
#   - Save results to file

from solution_demonstration.demonstration import Demonstration

# Compute demonstration results for CODAIT predictions.
# Then compute reconstruction errors using the developed autoencoder model.
demonstration = Demonstration(autoencoder_model_name="model_autoencoder_thesis.keras")
demonstration.codait_result_for_images(count=100, recursive=True)

# Save .csv data of all result sets
demonstration.save_normal_images_as_csv()
demonstration.save_mitosis_images_as_csv()
demonstration.save_all_images_as_csv()
demonstration.save_adversarial_images_as_csv()

# Print to console for debugging & evaluation
demonstration.print_average_prediction_error(
    label="normal", arr=demonstration.normal_images_predictions
)
demonstration.print_average_prediction_error(
    label="mitosis", arr=demonstration.mitosis_images_predictions
)
demonstration.print_average_prediction_error(
    label="both", arr=demonstration.all_images_predictions
)

# Print summaries of attack performance to file
demonstration.print_summary(
    title="CODAIT Prediction change (mitosis-to-normal)     ",
    before_data=demonstration.mitosis_images_predictions,
    after_data=demonstration.adv_mitosis_images_predictions,
    print_to_file=True,
)

demonstration.print_summary(
    title="CODAIT Prediction change (normal-to-mitosis)     ",
    before_data=demonstration.normal_images_predictions,
    after_data=demonstration.adv_normal_images_predictions,
    print_to_file=True,
)

demonstration.print_summary(
    title="Autoencoder reconstruction error (before-after)",
    before_data=demonstration.all_images_reconstruction_error,
    after_data=demonstration.all_adv_images_reconstruction_error,
    print_to_file=True,
)

# Visualize attack before & after results with box plots.
demonstration.boxplot_results(
    demonstration.mitosis_images_predictions,
    demonstration.adv_mitosis_images_predictions,
    label1="initial prediction",
    label2="prediction after attack",
    title="CODAIT Prediction change (mitosis-to-normal)",
)

demonstration.boxplot_results(
    demonstration.normal_images_predictions,
    demonstration.adv_normal_images_predictions,
    label1="initial prediction",
    label2="prediction after attack",
    title="CODAIT Prediction change (normal-to-mitosis)",
)

demonstration.boxplot_results(
    demonstration.all_images_reconstruction_error,
    demonstration.all_adv_images_reconstruction_error,
    label1="Initial reconstruction error",
    label2="Reconstruction error after attack",
    title="Autoencoder reconstruction error (before-after)",
)
