from src.attack.attack_sample_creator import AdversarialImagePatchCreator

# Create AdversarialImagePatchCreator
adv_patch_creator = AdversarialImagePatchCreator(basepath="src/dataset", verbose=False)
adv_patch_creator.expected_change_factor_mitosis = 100

# Create adversarial images from mitosis patches
adv_patch_creator.generate_adversarial_image_patches_for_mitosis_imageset(50)

# Show overall stats:
adv_patch_creator.overall_attack_performance()

# Save overall stats to file:
adv_patch_creator.attack_performance_to_file()
