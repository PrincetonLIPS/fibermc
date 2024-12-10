import functools 
from typing import List, Tuple

from jax import jit, vmap
import jax.numpy as np 
import jax.random as npr 
import numpy as static_np 

from fibers.estimators import sample_fibers
from constants import FLOAT_TYPE

def make_batch_idxs(num_elements: int, batch_size: int) -> list: 
    """Constructs a list of batches of size `batch_size`. Each 
    batch is itself an 1D integer-valued np.ndarray of length 
    `batch_size`.

    Parameters
    ----------
    num_elements: int 
        total number of elements to construct the batches from. 
    batch_size: int 
        integer-valued size of each batch to construct. 

    Examples
    --------
    >>> make_batch_idxs(9, 3) 
    >>> [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    >>> make_batch_idxs(9, 2)
    >> [[0, 1], [2, 3], [4, 5], [6, 7], [8]]

    Returns 
    -------
    batches: List[np.ndarray]
        list of integer-valued `np.np.ndarray`s comprising each 'batch'. 
    """
    assert batch_size <= num_elements, f"batch size {batch_size} cannot be larger than the total number of inputs {num_elements}"
    assert batch_size > 0, f"batch size {batch_size} must be a positive integer "

    # --- construct the list of 'full' (size `batch_size`) batches 
    num_full_batches: int = num_elements // batch_size
    full_batches: List[np.ndarray] = [np.arange(i * batch_size, (i + 1) * batch_size) for i in range(num_full_batches)]

    # --- (optionally) add a last non-'full' batch (see example 2 in the docstring) 
    batches: List[np.ndarray] = full_batches if ((num_elements % batch_size) == 0) else (full_batches + [np.arange(num_full_batches * batch_size, num_elements)])
    return batches 

@functools.partial(jit, static_argnums=(1,))
def translation_from_id(tile_id: np.ndarray, tile_dimension: int) -> np.ndarray: 
    """Computes the translation vector associated with the provided tile 
    identifier and the dimension of each tile. 

    Parameters
    ----------
    tile_id: np.ndarray
        np.ndarray of length 2 which contains a row and column index, respectively, 
        to identify the location of the tile of interest. 
    tile_dimension: int
        dimension of each tile (assumed to be equivalent along both rows and columns). 

    Returns
    -------
    translation: np.ndarray
        2D array representing the translation vector applied to an object to place 
        it in the bounds of the tile of interest. 
    """
    tile_row, tile_column = tile_id 
    translation: np.ndarray = np.array([
        tile_column * tile_dimension, 
        tile_row * tile_dimension
        ])
    return translation 


def translate_fibers(fibers: np.ndarray, translation: np.ndarray) -> np.ndarray: 
    """TODO don't need this method? 
    """
    fibers: np.ndarray = fibers + translation 
    return fibers 

def compute_tile_ids(target_shape: np.ndarray, tile_dimension: int) -> np.ndarray: 
    """compute_tile_ids.

    Parameters
    ----------
    target_shape: np.ndarray
        shape of the target array (i.e., the size of the array to be rendered downstream 
        from this utility). 
    tile_dimension: int
        dimension of each tile (assumed to be equivalent along both rows and columns). 

    Returns
    -------
    tile_ids: np.ndarray
        np.np.ndarray of shape (target_rows * target_columns, 2) containing each 2D 
        tile identifier along the second axis (i.e., axis=1). 
    """
    # --- ensure the target shape is divided by the tile dimension 
    target_rows, target_columns = target_shape[:2]
    #assert divides(target_rows, tile_dimension), f"target number of rows {target_rows} does not divide tile dimension {tile_dimension}"
    #assert divides(target_columns, tile_dimension), f"target number of columns {target_columns} does not divide tile dimension {tile_dimension}"

    num_tiles_x: int = target_columns // tile_dimension
    num_tiles_y: int = target_rows // tile_dimension

    tile_ids: np.ndarray = np.dstack(np.meshgrid(np.arange(num_tiles_x), np.arange(num_tiles_y))).reshape(-1, 2)
    return tile_ids

def get_bounds_from_hull(pixel_hull: np.ndarray) -> np.ndarray: 
    return np.array([
        pixel_hull[:, 0].min(), 
        pixel_hull[:, 1].min(), 
        pixel_hull[:, 0].max(), 
        pixel_hull[:, 1].max(), 
        ])

def sample_perfect(fibers_per_pixel: int, pixel_hulls: np.ndarray) -> np.ndarray: 
    num_pixels: int = pixel_hulls.shape[0]
    bounds: np.ndarray = vmap(get_bounds_from_hull)(pixel_hulls)

    key: np.ndarray = npr.PRNGKey(0)
    keys: np.ndarray = npr.split(key, num_pixels) 
    fibers, _ = vmap(sample_fibers, in_axes=(0, 0, None, None))(keys, bounds, fibers_per_pixel, 0.5)
    return np.squeeze(fibers)

def setup_tiles(args, target_resolution: np.ndarray) -> tuple: 
    """Configure an np.ndarray of fibers (to be shared among each tile by 
    translation downstream) and an np.ndarray of tile identifiers given 
    some configuration data and a target rendering resolution. 

    Parameters
    ----------
    args : namespace
        TODO replace with config dataclass 
    target_resolution : np.ndarray
        resolution of the rendering target. 

    Returns
    -------
    tile_data (fibers, tile_ids) : Tuple[np.ndarray, np.ndarray]
        Tuple comprised of an array of fibers and an array of tile identifiers. 
    """
    # --- determine (spatial) boundary of each tile 
    tile_bounds: tuple = (args.tile_dimension, args.tile_dimension) 

    # --- compute tile identifiers 
    tile_ids: np.ndarray = static_np.array(compute_tile_ids(target_resolution, args.tile_dimension))
    num_tiles: int = tile_ids.shape[0]
    args.log.info(f"using {num_tiles} tiles of dimension: ({args.tile_dimension}, {args.tile_dimension})")

    # --- sample fibers in 0th tile to be shared (via translation) among the other tiles
    fibers: np.ndarray = np.array(sample_fibers(npr.PRNGKey(0), args.num_fibers, args.fiber_length, tile_bounds)) 
    args.log.info(f"sampled {args.num_fibers} fibers ({human_bytes_str(fibers.nbytes)})")
    return fibers, tile_ids

def compute_background_area(image: np.ndarray, mask: np.ndarray) -> float:
    """Compute the total area of the background region associated with an image,
    when the image is projected onto the unit square [0, 1] x [0, 1].

    Parameters
    ----------
    image: np.ndarray 
        array containing an image; currently assumed to be square (i.e. image.shape[0] == image.shape[1]). 
    mask: np.ndarray 
        integer or boolean-valued array with a 1 (True) at indices where the image 
        is background and a 0 (False) at indices where the image is foreground. 

    Example
    -------
    >>> compute_background_area(np.zeros((2, 2)), np.eye(2))
    >>> 0.5

    Returns 
    -------
    background_area: float 
        proportion of the image that is background (computed as the product of the area of a 
        pixel and the number of background pixels). 
    """
    num_rows, num_columns, num_channels = image.shape
    assert num_rows == num_columns, f"image with {num_rows} rows and {num_columns} columns is not square (non-square images are not supported)."

    pixel_area: float = (1 / num_rows) ** 2
    num_background_pixels: int = np.sum(mask)
    background_area: float = pixel_area * num_background_pixels
    return background_area


def compute_background_pixel_hulls(image: np.ndarray, mask: np.ndarray) -> list:
    """Instantiates convex hulls (represented as arrays of line segments) for each pixel 
    contained in the background of the image, which is identified with a value of `1.`` in
    the `mask` array at the corresponding location.

    Parameters
    ----------
    image: np.ndarray 
        array containing an image; currently assumed to be square (i.e. image.shape[0] == image.shape[1]). 
    mask: np.ndarray 
        integer or boolean-valued array with a 1 (True) at indices where the image 
        is background and a 0 (False) at indices where the image is foreground. 
        
    Returns 
    -------
    hulls: List[np.ndarray]
        list of convex hulls for each background pixel (that is, the length of this list 
        is equal to np.sum(mask)); each hull is represented as a (4, 2) array of line segments 
        oriented counter-clockwise from the bottom-right hand corner. 
    """
    num_rows, num_columns, _ = image.shape
    assert num_rows == num_columns, f"image with {num_rows} rows and {num_columns} columns is not square (non-square images are not supported)."

    pixel_side_length: float = 1 / num_rows
    background_indices: np.ndarray = np.array(np.nonzero(mask)).T
    hulls: List[np.ndarray] = []

    for x, y in background_indices:
        hull: np.ndarray = np.array(
            [
                [(x + 1) * pixel_side_length, y * pixel_side_length],           # bottom right-hand corner
                [x * pixel_side_length, y * pixel_side_length],                 # bottom left-hand corner 
                [x * pixel_side_length, (y + 1) * pixel_side_length],           # upper left-hand corner 
                [(x + 1) * pixel_side_length, (y + 1) * pixel_side_length],     # upper right-hand corner
            ]
        )
        hull_counter_clockwise: np.ndarray = hull[::-1]
        hulls.append(hull_counter_clockwise)

    return hulls

def create_pixel_hull(pixel_coordinate: np.ndarray, ccw: bool = True, dtype=FLOAT_TYPE) -> np.ndarray:
    """Create a pixel hull (i.e., a (4, 2) array with the vertices associated with a 
    square convex hull) given a coordinate representing its location. 

    Parameters
    ----------
    pixel_coordinate: np.ndarray 
        a length 2 array encoding the location (in terms of row/column) of the 
        pixel for which to create a hull. 
    ccw: bool 
        whether to orient the hull counter-clockwise (default: True). 
    dtype: type 
        numeric type for the values comprising the resultant pixel hull (default: FLOAT_TYPE). 

    Returns 
    -------
    pixel_hull: np.ndarray 
        array containing 4 vertices of length 2 comprising the convex hull of the 
        pixel. 
    """
    pixel_hull_clockwise: np.ndarray = np.array(
        [
            [pixel_coordinate[0] + 1, pixel_coordinate[1]],
            [pixel_coordinate[0], pixel_coordinate[1]],
            [pixel_coordinate[0], pixel_coordinate[1] + 1],
            [pixel_coordinate[0] + 1, pixel_coordinate[1] + 1],
        ],
        dtype=dtype,
    )

    if ccw:
        pixel_hull: np.ndarray = pixel_hull_clockwise[::-1]
    else: 
        pixel_hull: np.ndarray = pixel_hull_clockwise

    return pixel_hull

def random_image(size: Tuple[int]) -> np.ndarray: 
    """Generate a psuedo-random image (the model is uniform over the 
    color-space) of size `size`.
    
    Parameters
    ----------
    size: Tuple[int] 
        tuple of integers representing the desired generated image shape.

    Returns
    -------
    image: np.ndarray 
        floating-point valued np.ndarray with values ranging between 0.0 and 1.0 
        sampled according to a uniform distribution. 
    """
    image: np.ndarray = static_np.random.uniform(size=(size))
    return image 

@functools.partial(jit, static_argnums=(0, 1))
def get_pixel_coordinates(width: int, height: int, bounds: np.ndarray) -> np.ndarray: 
    """Returns an array containing the elements of the Cartesian product 
    of the sets of integers comprising the range of rows (0, 1, ..., size[0]-1) and 
    columns (0, 1, ..., size[1]-1) given. 

    Parameters
    ----------
    bounds: ndaray
        np.ndarray of integers representing the associated image shape. 

    Example
    -------
    >>> get_pixel_coordinates(np.array([0, 0, 2, 2]))
    >>> array([0, 0], [0, 1], [1, 0], [1, 1])

    Returns 
    -------
    pixel_coordinates: np.ndarray 
        array of shape (np.prod(size), 2) containing all the length 2 pixel 
        coordinates (row, column) associated with an image of size `size`. 
    """
    min_x, min_y, _, _ = bounds[:4]
    x_coords: np.ndarray = np.arange(width) + min_x 
    y_coords: np.ndarray = np.arange(height) + min_y
    coordinates: np.ndarray = np.dstack(np.meshgrid(x_coords, y_coords)).reshape(-1, 2)
    return coordinates 


def compute_vertex_mask(vertices: np.ndarray) -> np.ndarray: 
    """Constructs a binary mask array to prevent updates on 'boundary' vertices 
    present in the input `vertices` array. 

    Parameters
    ----------
    vertices: np.ndarray 
        (N, 2) array of vertex positions. 

    Returns 
    -------
    vertex_mask: np.ndarray
        binary integer-valued {0, 1} array of length (N, 1) encoding whether, for 
        each of the N vertices in the input array, whether that vertex should be 
        held fixed (because it is at the boundary). 
    """
    # --- extract coordinate components 
    vertex_x_coordinates: np.ndarray = vertices[:, 0] 
    vertex_y_coordinates: np.ndarray = vertices[:, 1] 

    # --- determine boundary coordinates 
    min_x_coordinate: np.ndarray = vertices[:, 0].min() 
    max_x_coordinate: np.ndarray = vertices[:, 0].max() 

    min_y_coordinate: np.ndarray = vertices[:, 1].min() 
    max_y_coordinate: np.ndarray = vertices[:, 1].max() 

    # --- construct the mask 
    x_mask: np.ndarray = static_np.logical_or(vertex_x_coordinates == min_x_coordinate, vertex_x_coordinates == max_x_coordinate)
    y_mask: np.ndarray = static_np.logical_or(vertex_y_coordinates == min_y_coordinate, vertex_y_coordinates == max_y_coordinate)
    full_mask: np.ndarray = ~static_np.logical_or(x_mask, y_mask)
    vertex_mask: np.ndarray = full_mask.astype(int).reshape(-1, 1)
    return vertex_mask

def apply_vertex_mask(gradient: tuple, mask: np.ndarray) -> tuple: 
    """TODO need this function? 
    """
    return (gradient[0], gradient[1] * mask)

def apply_vertex_clip(bounds: tuple, params: tuple) -> tuple: 
    """TODO don't take params, just take vertices? 
    """
    vertices: np.ndarray = params[1] 

    min_x, min_y, max_x, max_y = bounds 

    vertex_x: np.ndarray = np.clip(vertices[:, 0], a_min=min_x, a_max=max_x) 
    vertex_y: np.ndarray = np.clip(vertices[:, 1], a_min=min_y, a_max=max_y) 

    clipped_vertices: np.ndarray = np.stack((vertex_x, vertex_y)).T

    # TODO this is hacky? 
    if len(params) == 2: 
        return (params[0], clipped_vertices)
    elif len(params) == 3: 
        return (params[0], clipped_vertices, params[2])

def _apply_vertex_clip(bounds: tuple, vertices: np.ndarray) -> tuple: 
    """TODO don't take params, just take vertices? 
    """
    min_x, min_y, max_x, max_y = bounds 

    vertex_x: np.ndarray = np.clip(vertices[:, 0], a_min=min_x, a_max=max_x) 
    vertex_y: np.ndarray = np.clip(vertices[:, 1], a_min=min_y, a_max=max_y) 

    clipped_vertices: np.ndarray = np.stack((vertex_x, vertex_y)).T
    return clipped_vertices

def bounds_from_tile_id(tile_id: np.ndarray, tile_dimension: int) -> np.ndarray: 
    """Computes an np.ndarray parameterizing a rectangular boundary, given a 
    tile identifier and a dimension (measured in pixels) of each tile. 
    """
    tile_x, tile_y = tile_id
    return np.array([tile_x * tile_dimension, tile_y * tile_dimension, (tile_x + 1) * tile_dimension, (tile_y + 1) * tile_dimension])

def get_device_batches(config) -> np.ndarray: 
    # --- ensure the requested devices are available 
    num_devices_available: int = get_num_devices()
    requested_devices: int = config.compute.multi_gpu.num_devices
    assert requested_devices <= num_devices_available, f"requested {requested_devices} devices but only {num_devices_available} available."
    num_devices: int = requested_devices

    # --- if there are fewer tiles than devices, only use one device per tile
    num_tiles: int = config.tile_ids.shape[0]
    num_devices: int = min(num_tiles, num_devices)
    args.log.info(f"using {num_devices} of {num_devices_available} available devices")

    # --- aggregate batch indices of tiles for launching on devices TODO generalize 
    assert num_tiles % num_devices == 0, "number of devices does not equally divide number of tiles"
    batch_idxs: np.ndarray = np.arange(num_tiles).reshape(num_devices, int((num_tiles / num_devices)))
    device_batches: np.ndarray = config.tile_ids[batch_idxs, :]
    return device_batches

def get_tile(target_image: np.ndarray, tile_dimension: int, tile_id: np.ndarray) -> np.ndarray: 
    x_start, y_start = tile_id 
    target_shape: np.ndarray = np.array(target_image.shape[:2])
    num_pixels: int = np.prod(target_shape) 
    pixels_per_tile: int = tile_dimension ** 2 
    tile: np.ndarray = np.zeros((tile_dimension, tile_dimension, 3)).astype(np.float32)
    target: np.ndarray = static_np.array(target_image)

    tile_x_start: int = x_start * tile_dimension 
    tile_y_start: int = y_start * tile_dimension 
    tile_x_end: int = tile_x_start + tile_dimension
    tile_y_end: int = tile_y_start + tile_dimension

    tile = np.array(target[tile_x_start:tile_x_end, tile_y_start:tile_y_end, :])
    return tile 

def compute_tile_power(target_image: np.ndarray, tile_dimension: int, tile_id: np.ndarray) -> float:
    tile: np.ndarray = get_tile(target_image, tile_dimension, tile_id)
    power_spectrum: np.ndarray = (np.abs(np.fft.fftn(tile))**2).ravel()
    n: int = power_spectrum.size // 2
    power: float = power_spectrum[:n].dot(np.arange(n))
    return power

def compute_tilewise_sparsity(target_image: np.ndarray, tile_dimension: int) -> np.ndarray: 
    tile_ids: np.ndarray = compute_tile_ids(target_image.shape, tile_dimension) 
    tile_powers: np.ndarray = np.array([functools.partial(compute_tile_power, target_image, tile_dimension)(tile_id) for tile_id in tile_ids])
    tile_powers: np.ndarray = tile_powers / tile_powers.sum() 
    return 1. - tile_powers 

def rescale_sparsity(tilewise_sparsity: np.ndarray, min_sparsity: float, max_sparsity: float) -> np.ndarray: 
    min_power: float = tilewise_sparsity.min()
    max_power: float = tilewise_sparsity.max() 
    rescaled: np.ndarray = (max_sparsity - min_sparsity) * ((tilewise_sparsity - min_power)/(max_power - min_power)) + min_sparsity 
    return rescaled 
