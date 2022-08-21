import json

training_parameters = {
    # stage-1 contrastive learning training details
    "contrastive_learning_model": [{
        'input_image_size': (256, 256, 3),
        'output_vector_size': (1, 128),
        'epochs': 50,
        'batch_size': 8,
        'optimizer': [{
            'name': 'Adam',
            'learning_rate': 0.001,
            'beta_1': 0.9,
            'beta_2': 0.99
        }],
    }],
    # stage-2 conditional variational autoencoder details
    "cvae_model": [{
        'input_image_size': (256, 256, 3),
        'input_conditional_vector_size': (1, 128),
        'output_image_size': (256, 256, 3),
        'latent_dimension_size': (1, 128),
        'epochs': 150,
        'batch_size': 4,
        'optimizer': [{
            'name': 'Adam',
            'learning_rate': 0.001,
            'beta_1': 0.9,
            'beta_2': 0.99
        }],
    }]
}
