from clothing_segmentation.preprocessing.indian_guys import segment_garment

segment_garment(inputs_dir='/home/roman/tryondiffusion_implementation/assets/inputs_dir_segment_garment',
               outputs_dir='/home/roman/tryondiffusion_implementation/assets/ouptuts_dir_segment_garment', cls='all')

# from tryon.preprocessing import extract_garment

# extract_garment(inputs_dir='/home/roman/tryondiffusion_implementation/assets/inputs_dir_segment_garment',
#                outputs_dir='/home/roman/tryondiffusion_implementation/assets/ouptuts_dir_extract_garment', cls='all')

# from tryon.preprocessing import segment_human

# segment_human(image_path='/home/roman/tryondiffusion_implementation/assets/inputs_dir_segment_garment/image-1.png',
#                output_dir='/home/roman/tryondiffusion_implementation/assets/ouptuts_dir_extract_garment')


# (venv_cloth_segm) roman@ai-gen1:~/tryondiffusion_implementation/tryondiffusion_tryonlabs$ PYTHONPATH=. python3 tryon/test_clothing_segmentation.py 