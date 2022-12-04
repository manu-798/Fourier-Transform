import numpy as np
import matplotlib.pyplot as plt

image_file = "vlcsnap.png"

def fourier_transform(input):
    ft = np.fft.ifftshift(input)
    ft = np.fft.fft2(ft)
    return np.fft.fftshift(ft)

def inverse_fourier_transform(input):
    ift = np.fft.ifftshift(input)
    ift = np.fft.ifft2(ift)
    ift = np.fft.fftshift(ift)
    return ift.real

def calculate_distance_from_centre(coords, centre):
    return np.sqrt(
        (coords[0] - centre) ** 2 + (coords[1] - centre) ** 2
    )

def find_symmetric_coordinates(coords, centre):
    return (centre + (centre - coords[0]),
            centre + (centre - coords[1]))

def display_plots(individual_grating, reconstruction, idx):
    plt.subplot(121)
    plt.imshow(individual_grating)
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(reconstruction)
    plt.axis("off")
    plt.suptitle(f"Terms: {idx}")
    plt.pause(0.01)

image = plt.imread(image_file)
image = image[:, :, :3].mean(axis=2)  

array_size = min(image.shape) - 1 + min(image.shape) % 2

image = image[:array_size, :array_size]
centre = int((array_size - 1) / 2)

coords_left_half = (
    (x, y) for x in range(array_size) for y in range(centre+1)
)

coords_left_half = sorted(
    coords_left_half,
    key=lambda x: calculate_distance_from_centre(x, centre)
)

plt.set_cmap("gray")

ft = fourier_transform(image)

plt.subplot(121)
plt.imshow(image)
plt.axis("off")
plt.subplot(122)
plt.imshow(np.log(abs(ft)))
plt.axis("off")
plt.pause(2)


fig = plt.figure()

rec_image = np.zeros(image.shape)
individual_grating = np.zeros(
    image.shape, dtype="complex"
)
idx = 0

display_all_until = 200

display_step = 10

next_display = display_all_until + display_step

for coords in coords_left_half:

    if not (coords[1] == centre and coords[0] > centre):
        idx += 1
        symm_coords = find_symmetric_coordinates(
            coords, centre
        )

        individual_grating[coords] = ft[coords]
        individual_grating[symm_coords] = ft[symm_coords]


        rec_grating = inverse_fourier_transform(individual_grating)
        rec_image += rec_grating

        individual_grating[coords] = 0
        individual_grating[symm_coords] = 0

        if idx < display_all_until or idx == next_display:
            if idx > display_all_until:
                next_display += display_step
                display_step += 10
            display_plots(rec_grating, rec_image, idx)

plt.show()
