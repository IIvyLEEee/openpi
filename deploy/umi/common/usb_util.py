import fcntl
import os
import pathlib
from subprocess import DEVNULL, PIPE, Popen


def create_usb_list():
    device_list = []
    lsusb_out = Popen(
        "lsusb -v",
        shell=True,
        bufsize=64,
        stdin=PIPE,
        stdout=PIPE,
        stderr=DEVNULL,
        close_fds=True,
    ).stdout.read().strip().decode("utf-8")

    usb_devices = lsusb_out.split(f"{os.linesep}{os.linesep}")
    for device_categories in usb_devices:
        if not device_categories:
            continue
        categories = device_categories.split(os.linesep)
        device_stuff = categories[0].strip().split()
        bus = device_stuff[1]
        device = device_stuff[3][:-1]
        device_dict = {"bus": bus, "device": device}
        device_info = " ".join(device_stuff[6:])
        device_dict["description"] = device_info
        for category in categories:
            if not category:
                continue
            category_info = category.strip().split()
            if category_info[0] == "iManufacturer":
                manufacturer_info = " ".join(category_info[2:])
                device_dict["manufacturer"] = manufacturer_info
            if category_info[0] == "iProduct":
                device_info = " ".join(category_info[2:])
                device_dict["device"] = device_info
        device_dict["path"] = f"/dev/bus/usb/{bus}/{device}"
        device_list.append(device_dict)
    return device_list


def reset_usb_device(dev_path):
    USBDEVFS_RESET = 21780
    try:
        with open(dev_path, "w", os.O_WRONLY) as f:
            fcntl.ioctl(f, USBDEVFS_RESET, 0)
        print(f"Successfully reset {dev_path}")
    except PermissionError as ex:
        raise PermissionError(f'Try running "sudo chmod 777 {dev_path}"') from ex


def reset_all_elgato_devices():
    """
    Find and reset all Elgato capture cards.
    Required to work around a firmware bug, matching UMI's UVC startup path.
    """
    for dev in create_usb_list():
        if "Elgato" in dev.get("description", ""):
            reset_usb_device(dev["path"])


def get_sorted_v4l_paths(by_id=True):
    """Return stable V4L2 camera device paths for UVC/GoPro capture cards."""
    dirname = "by-id" if by_id else "by-path"
    v4l_dir = pathlib.Path("/dev/v4l").joinpath(dirname)

    valid_paths = []
    for dev_path in sorted(v4l_dir.glob("*video*")):
        name = dev_path.name
        index_str = name.split("-")[-1]
        assert index_str.startswith("index")
        index = int(index_str[5:])
        if index == 0:
            valid_paths.append(dev_path)

    if valid_paths:
        return [str(x.absolute()) for x in valid_paths]

    return [str(x.absolute()) for x in sorted(pathlib.Path("/dev").glob("video*"))]
