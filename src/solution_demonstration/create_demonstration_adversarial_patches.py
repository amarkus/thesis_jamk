from src.attack.attack_sample_creator import AdversarialImagePatchCreator

# Create AdversarialImagePatchCreator
adv_patch_creator = AdversarialImagePatchCreator(
    basepath="src/dataset",
    mitosis_images_path="demonstration_patches/original",
    adversarial_images_path="demonstration_patches/adversarial",
    attack_logs_path="src/reports/black_box_attacks",
    verbose=False,
)

adv_patch_creator.expected_change_factor_mitosis = 100
adv_patch_creator.expected_change_factor_normal = 80
original_path = adv_patch_creator.mitosis_images_path

# Create adversarial images from demonstration dataset
adv_patch_creator.generate_adversarial_patch_for_dataset(
    images_path=original_path, count=100
)

# Show overall stats:
adv_patch_creator.overall_attack_performance()

# Save overall stats to file:
adv_patch_creator.attack_performance_to_file()
