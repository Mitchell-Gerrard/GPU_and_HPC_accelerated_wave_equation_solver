import pyopencl as cl

for platform in cl.get_platforms():
    print("PLATFORM:", platform.name)

    for device in platform.get_devices():
        print("DEVICE:", device.name)
        print("TYPE:", cl.device_type.to_string(device.type))
        print("GLOBAL MEMORY:", device.global_mem_size / 1e9, "GB")
        print("MAX WORK GROUP:", device.max_work_group_size)
        print("COMPUTE UNITS:", device.max_compute_units)
        print()