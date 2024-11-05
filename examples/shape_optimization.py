import jax 
import jax.numpy as np 
import jax.random as npr
import matplotlib.pyplot as plt 

import estimators
import splines 

def show_configuration(fields: tuple[callable], s1_controls: np.ndarray, s2_controls: np.ndarray, domain_bounds: np.ndarray, **kwargs) -> None: 
    f, union_field, intersection_field = fields

    resolution: int = kwargs.get("resolution", 750)
    _x: np.ndarray = np.linspace(domain_bounds[0], domain_bounds[2], num=resolution)
    _y: np.ndarray = np.linspace(domain_bounds[1], domain_bounds[3], num=resolution)
    _X, _Y = np.meshgrid(_x, _y)
    inputs = np.dstack((_X, _Y)).reshape(-1, 2)
    Z1: np.ndarray = jax.vmap(lambda x: f(s1_controls, x))(inputs).reshape(resolution, resolution)
    Z2: np.ndarray = jax.vmap(lambda x: f(s2_controls, x))(inputs).reshape(resolution, resolution)

    U1: np.ndarray = jax.vmap(lambda x: union_field(s2_controls, x))(inputs).reshape(resolution, resolution)
    I1: np.ndarray = jax.vmap(lambda x: intersection_field(s2_controls, x))(inputs).reshape(resolution, resolution)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    axs[0].set_title("Union")
    axs[0].contour(Z1 < 0, [0.], colors="tab:blue")
    axs[0].contour(Z2 < 0, [0.], colors="tab:green")
    axs[0].contourf(I1 < 0, [-0.1, 0.1], colors="tab:blue", alpha=0.1)
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    axs[1].set_title("Intersection")
    axs[1].contour(Z1 < 0, [0.], colors="tab:blue")
    axs[1].contour(Z2 < 0, [0.], colors="tab:green")
    axs[1].contour(U1 < 0, [0.], colors="tab:red")
    axs[1].contourf(U1 < 0, [-0.1, 0.1], colors="tab:red", alpha=0.1)
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    if kwargs.get("save_path", None) is not None: 
        plt.savefig(kwargs["save_path"])
        plt.close()
    else: 
        plt.show()


def main(): 
    # configuration 
    domain_degree: int = 2
    domain_height: float = 1. 
    domain_width: float = 1. 
    num_control_points: int = 25
    shape_radius: float = 0.3

    domain_bounds: np.ndarray = np.array([
        0.,                     # min x
        0.,                     # min y
        domain_height,          # max x  
        domain_width,           # max y  
    ])

    # initial control points 
    _x: np.ndarray = np.linspace(domain_bounds[0], domain_bounds[2], num=num_control_points)
    _y: np.ndarray = np.linspace(domain_bounds[1], domain_bounds[3], num=num_control_points)
    domain_inputs: np.ndarray = np.stack(np.meshgrid(_x, _y), axis=-1).reshape(-1, 2)

    # shapes
    s1_center: np.ndarray = np.array([0.5, 0.35])
    s1_controls: np.ndarray = jax.vmap(lambda x: -(np.linalg.norm(x - s1_center, ord=1) - shape_radius))(domain_inputs).reshape(num_control_points, num_control_points)

    s2_center: np.ndarray = np.array([0.5, 0.65])
    s2_controls: np.ndarray = jax.vmap(lambda x: -(np.linalg.norm(x - s2_center, ord=1) - shape_radius))(domain_inputs).reshape(num_control_points, num_control_points)

    # knots 
    knots: np.ndarray = splines.default_knots(domain_degree, num_control_points)
    knots = (knots, knots)
    
    # spline implicit function 
    def f(controls: np.ndarray, x: np.ndarray) -> np.ndarray: 
        x = x.reshape(1, -1)
        x /= np.array([domain_height, domain_width])
        return splines.bspline2d(x, controls, knots, domain_degree).squeeze()

    def union_field(s2_controls: np.ndarray, x: np.ndarray): 
        return np.min(np.array([f(s1_controls, x), f(s2_controls, x)]))

    def intersection_field(s2_controls: np.ndarray, x: np.ndarray): 
        return np.max(np.array([f(s1_controls, x), f(s2_controls, x)]))

    fields = (f, union_field, intersection_field)

    # basic IOU objective (with one shape fixed)
    def objective(controls: np.ndarray, fibers: np.ndarray) -> np.ndarray: 
        union_area: np.ndarray = estimators.estimate_field_area(union_field, fibers, controls)
        interection_area: np.ndarray = estimators.estimate_field_area(intersection_field, fibers, controls)
        iou: np.ndarray = (interection_area / union_area).squeeze()
        return -iou, (interection_area, union_area)

    # fibers
    key: np.ndarray = npr.PRNGKey(0)
    num_fibers: int = 500
    fiber_length: float = 5e-2

    num_steps: int = 750 
    report_every: int = 25
    plot_every: int = 100
    step_size: float = 5e-01
    gradient_fn: callable = jax.jit(jax.value_and_grad(objective, has_aux=True))

    save_path = f"configuration_init.png"
    show_configuration(fields, s1_controls, s2_controls, domain_bounds, save_path=save_path)

    for i in range(num_steps): 
        key, next_key = npr.split(key)
        fibers: np.ndarray = jax.jit(estimators.sample, static_argnums=(2, 3))(key, domain_bounds, num_fibers, fiber_length)
        key = next_key 

        if i == 0: 
            print(f"Compiling...")

        (current_objective, aux), gradient= gradient_fn(s2_controls, fibers)
        intersection_area, union_area = aux 
        s2_controls -= gradient * step_size

        if i == 0: 
            print(f"Compiled...")

        if i % report_every == 0:
            print(f"Iteration [{i:04d}/{num_steps:04d}]\tIOU: {-current_objective.item():0.2f}\tIntersection: {intersection_area.item():0.3f}\tUnion: {union_area.item():0.3f}")

        if i % plot_every == 0: 
            save_path = f"configuration_{i}.png"
            show_configuration(fields, s1_controls, s2_controls, domain_bounds, save_path=save_path)

    save_path = f"configuration_final.png"
    show_configuration(fields, s1_controls, s2_controls, domain_bounds, save_path=save_path)

if __name__=="__main__": 
    main()