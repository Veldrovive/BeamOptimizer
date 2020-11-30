import numpy as np
from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize

def area(web_height, flange_width, glue_width, bridge_width, bridge_length, thickness, a):
    num_diaphragms = np.ceil(bridge_length / a)
    deck_area = bridge_length * bridge_width * 2
    diaphragm_area = num_diaphragms * web_height * (bridge_width - 2 * flange_width - 2 * thickness)
    glue_pad_area = 2 * bridge_length * glue_width
    web_area = 2 * bridge_length * (web_height - thickness)
    return deck_area + diaphragm_area + glue_pad_area + web_area

def get_max_force(web_height, flange_width, glue_width, a = 300):
    bridge_width = 120
    bridge_length = 950
    moment = 225
    shear = 0.75
    max_tensile_stress = 30
    max_compressive_stress = 6
    max_shear = 4
    max_glue_shear = 2
    E = 4000
    thickness = 1.27
    poisson = 0.2

    deck_center_width = bridge_width - 2 * flange_width
    deck_thickness = 2 * thickness
    web_thickness = thickness 

    # print("Thickness Params:", deck_center_width, deck_thickness, web_thickness)
    
    y_bar = ((bridge_width * deck_thickness * (web_height + deck_thickness / 2)) + (2 * web_thickness * web_height ** 2 / 2)) / (bridge_width * deck_thickness + 2 * web_thickness * web_height)  # TODO: Take glue pads into account
    y_top = web_height + deck_thickness - y_bar
    y_bot = y_bar
    deck_i = ((bridge_width * deck_thickness ** 3) / 12) + bridge_width * deck_thickness *(web_height + deck_thickness / 2 - y_bar)**2
    web_i = ((web_thickness * web_height ** 3) / 12) + web_thickness * web_height *(web_height / 2 - y_bar)**2
    i = deck_i + 2*web_i

    # print("Bending Params:", y_bar, y_top, y_bot, deck_i, web_i, i)

    web_above_neutral_axis = web_height - y_bar

    q_glue = bridge_width * deck_thickness * (web_height + deck_thickness / 2 - y_bar)
    q_neutral = 2 * web_above_neutral_axis * web_thickness * ((web_height - y_bar) / 2) + q_glue

    # print("Qs", q_neutral, q_glue)

    def get_tensile_failure_load(max_tensile_stress, i, moment, y_bot):
        return (max_tensile_stress * i) / (moment * y_bot)

    def get_crushing_failure_load(max_compressive_stress, i, moment, y_top):
        return (max_compressive_stress * i) / (moment * y_top)

    def get_shear_failure_load(max_shear, i, web_thickness, shear, q_neutral):
        return (2 * max_shear * web_thickness * i) / (shear * q_neutral)

    def get_glue_failure_load(max_glue_shear, i, glue_width, shear, q_glue):
        return (2 * max_glue_shear * glue_width * i) / (shear * q_glue)

    def get_two_restrained_buckling_load(E, poisson, deck_thickness, deck_center_width, moment, y_top, i):
        critical_stress = ((4 * np.pi**2 * E) / (12 * (1 - poisson**2))) * (deck_thickness / deck_center_width)**2
        return (critical_stress * i) / (moment * y_top)

    def get_one_restrained_buckling_load(E, poisson, deck_thickness, flange_width, moment, y_top, i):
        critical_stress = ((0.425 * np.pi**2 * E) / (12 * (1 - poisson**2))) * (deck_thickness / flange_width)**2
        return (critical_stress * i) / (moment * y_top)

    def get_web_buckling_load(E, poisson, web_thickness, web_above_neutral_axis, moment, i):
        critical_stress = ((6 * np.pi**2 * E) / (12 * (1 - poisson**2))) * (web_thickness / web_above_neutral_axis)**2
        return (critical_stress * i) / (moment * web_above_neutral_axis)

    def get_web_shear_load(E, poisson, web_thickness, web_height, i, shear, q_neutral):
        critical_shear = ((5 * np.pi**2 * E) / (12 * (1 - poisson**2))) * ((web_thickness / web_height)**2 + (web_thickness / a)**2)
        return (2 * critical_shear * i * web_thickness) / (shear * q_neutral)

    tensile_failure = get_tensile_failure_load(max_tensile_stress, i, moment, y_bot)
    crushing_failure = get_crushing_failure_load(max_compressive_stress, i, moment, y_top)
    shear_failure = get_shear_failure_load(max_shear, i, web_thickness, shear, q_neutral)
    glue_failure  = get_glue_failure_load(max_glue_shear, i, glue_width, shear, q_glue)
    two_restrained_buckling_failure = get_two_restrained_buckling_load(E, poisson, deck_thickness, deck_center_width, moment, y_top, i)
    one_restrained_buckling_failure = get_one_restrained_buckling_load(E, poisson, deck_thickness, flange_width, moment, y_top, i)
    web_buckling_failure = get_web_buckling_load(E, poisson, web_thickness, web_above_neutral_axis, moment, i)
    web_shear_failure = get_web_shear_load(E, poisson, web_thickness, web_height, i, shear, q_neutral)
    # print("Tensile:", tensile_failure, "   Crushing:", crushing_failure, "   Shear:", shear_failure, "   Glue:", glue_failure, "   Two:", two_restrained_buckling_failure, "   One:", one_restrained_buckling_failure, "   Web Buckle:", web_buckling_failure, "   Web Shear:", web_shear_failure)

    return min(tensile_failure, crushing_failure, shear_failure, glue_failure, two_restrained_buckling_failure, one_restrained_buckling_failure, web_buckling_failure, web_shear_failure)

def optimize(web_height_guess = 100, flange_guess = 20, glue_guess = 5, a_guess = 100, max_area = 813 * 1016):
    bridge_width = 120
    thickness = 1.27
    max_web_height = 250
    max_glue_width = 30
    def area_con(x):
        web_height, flange_width, glue_width, a = x
        return max_area - area(web_height, flange_width, glue_width, bridge_width, 950, thickness, a)
    def flange_con(x):
        web_height, flange_width, glue_width, a = x
        return bridge_width / 2 - 2 * glue_width - flange_width
    def web_con(x):
        web_height, flange_width, glue_width, a = x
        return web_height
    def web_con_2(x):
        web_height, flange_width, glue_width, a = x
        return max_web_height - web_height
    def glue_con(x):
        web_height, flange_width, glue_width, a = x
        return glue_width
    def glue_con_2(x):
        web_height, flange_width, glue_width, a = x
        return max_glue_width - glue_width
    def a_con(x):
        web_height, flange_width, glue_width, a = x
        return a - 20
    def optimize(x):
        web_height, flange_width, glue_width, a = x
        # print("Running optimize", web_height, flange_width, glue_width, get_max_force(web_height, flange_width, glue_width))
        return -1 * get_max_force(web_height, flange_width, glue_width, a)

    cons = [
        {'type': 'ineq', 'fun': area_con},
        {'type': 'ineq', 'fun': flange_con},
        {'type': 'ineq', 'fun': web_con},
        {'type': 'ineq', 'fun': web_con_2},
        {'type': 'ineq', 'fun': glue_con},
        {'type': 'ineq', 'fun': glue_con_2},
        {'type': 'ineq', 'fun': a_con}
    ]
    res = minimize(optimize, [web_height_guess, flange_guess, glue_guess, a_guess], constraints=cons, options={'verbose': 1})
    return res.x

def search(params, max_area, bridge_length):
    start_web_height, start_flange_width, start_glue_width, start_a = params
    best_web_height, best_flange_width, best_glue_width, best_a = optimize(start_web_height, start_flange_width, start_glue_width, start_a)
    best_max_force = get_max_force(best_web_height, best_flange_width, best_glue_width, best_a)
    best_area = area(best_web_height, best_flange_width, best_glue_width, 120, 950, 1.27, best_a)
    area_left = max_area - best_area
    best_num_diaphragms = np.ceil(bridge_length / best_a)
    return best_max_force, area_left, best_web_height, best_flange_width, best_glue_width, best_a, best_num_diaphragms

def search_area(web_height_mm, flange_width_mm, glue_width_mm, a_mm, max_area=813 * 1016, bridge_length=950):
    min_wh, max_wh, steps_wh = web_height_mm
    min_fw, max_fw, steps_fw = flange_width_mm
    min_gw, max_gw, steps_gw = glue_width_mm
    min_a, max_a, steps_a = a_mm

    wh_linspace = np.linspace(min_wh, max_wh, steps_wh)
    fw_linspace = np.linspace(min_fw, max_fw, steps_fw)
    gw_linspace = np.linspace(min_gw, max_gw, steps_gw)
    a_linspace = np.linspace(min_a, max_a, steps_a)

    best = None
    for wh in wh_linspace:
        for fw in fw_linspace:
            for gw in gw_linspace:
                for a in a_linspace:
                    print("Params:", wh, fw, gw, a)
                    best_max_force, area_left, best_web_height, best_flange_width, best_glue_width, best_a, best_num_diaphragms = search([wh, fw, gw, a], max_area, 950)
                    if best is None:
                        best = [best_max_force, [best_web_height, best_flange_width, best_glue_width, best_a]]
                        print("New Best:", best_max_force)
                    elif best_max_force > best[0]:
                        best = [best_max_force, [best_web_height, best_flange_width, best_glue_width, best_a]]
                        print("New Best:", best_max_force)
    return search(best[1], max_area, bridge_length)

if __name__ == "__main__":
    max_area = 813 * 1016
    start_web_height = 200
    start_flange_width = 20
    start_glue_width = 10
    start_a = 300
    start_max_force = get_max_force(start_web_height, start_flange_width, start_glue_width, start_a)
    start_area = area(start_web_height, start_flange_width, start_glue_width, 120, 950, 1.27, start_a)

    print(f"Start - Max Force: {start_max_force}   Area: {start_area}   Area Left: {max_area - start_area}")

    # best_web_height, best_flange_width, best_glue_width, best_a = optimize(start_web_height, start_flange_width, start_glue_width, start_a)
    # best_max_force = get_max_force(best_web_height, best_flange_width, best_glue_width, best_a)
    # best_area = area(best_web_height, best_flange_width, best_glue_width, 120, 950, 1.27, best_a)

    # print(f"End - Max Force: {best_max_force}   Area: {best_area}   Area Left: {max_area - best_area}")
    # num_diaphragms = np.ceil(950 / best_a)
    # print(f"Best a: {best_a}   Best Number of Diaphragms: {num_diaphragms}   Best Height: {best_web_height}   Best Flange Width: {best_flange_width}   Best Glue Width: {best_glue_width}")
    # best_max_force, area_left, best_web_height, best_flange_width, best_glue_width, best_a, best_num_diaphragms = search([start_web_height, start_flange_width, start_glue_width, start_a], max_area, 950)
    # print(f"End - Max Force: {best_max_force}   Area: {max_area - area_left}   Area Left: {area_left}")
    # print(f"Best a: {best_a}   Best Number of Diaphragms: {best_num_diaphragms}   Best Height: {best_web_height}   Best Flange Width: {best_flange_width}   Best Glue Width: {best_glue_width}")
    best_max_force, area_left, best_web_height, best_flange_width, best_glue_width, best_a, best_num_diaphragms = search_area([50, 200, 4], [5, 30, 4], [1, 11, 4], [10, 300, 4])
    print(f"End - Max Force: {best_max_force}   Area: {max_area - area_left}   Area Left: {area_left}")
    print(f"Best a: {best_a}   Best Number of Diaphragms: {best_num_diaphragms}   Best Height: {best_web_height}   Best Flange Width: {best_flange_width}   Best Glue Width: {best_glue_width}")
