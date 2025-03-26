import tensorflow as tf
import os

# Try to force GPU detection
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

print(f"TensorFlow version: {tf.__version__}")
print(f"CUDA Visible Devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not Set')}")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not Set')}")

# Check if TensorFlow can see the GPU
physical_devices = tf.config.list_physical_devices('GPU')
print(f"GPUs detected by TensorFlow: {physical_devices}")

if physical_devices:
    # Enable memory growth
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        
        # Run a test computation on GPU
        print("Running test computation on GPU...")
        with tf.device('/GPU:0'):
            a = tf.random.normal([5000, 5000])
            b = tf.random.normal([5000, 5000])
            c = tf.matmul(a, b)
            print(f"Result shape: {c.shape}")
        print("GPU computation successful!")
    except Exception as e:
        print(f"Error configuring GPU: {e}")
else:
    print("No GPUs detected by TensorFlow. Let's try some troubleshooting:")

    # Check if CUDA toolkit is installed
    import subprocess
    try:
        nvcc_version = subprocess.check_output(['nvcc', '--version']).decode()
        print(f"\nNVCC Version:\n{nvcc_version}")
    except:
        print("\nNVCC not found. CUDA toolkit may not be installed properly.")
    
    # Add additional debugging
    print("\nTensorFlow CUDA build info:")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"Built with GPU support: {tf.test.is_built_with_gpu_support()}")
    
    # Check for GPU/CUDA related libraries
    print("\nChecking for CUDA libraries:")
    cuda_so_files = subprocess.check_output(['ldconfig', '-p', '|', 'grep', 'cuda'], shell=True).decode()
    print(cuda_so_files[:500] + "..." if len(cuda_so_files) > 500 else cuda_so_files)
