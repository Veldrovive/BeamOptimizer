import os
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from rectpack import newPacker, float2dec
import warnings
warnings.filterwarnings("ignore")

from typing import List, Tuple

def squeeze(val, mi, ma):
    """
    If val < mi or val > ma, returns mi or ma respectively
    """
    return min(ma, max(val, mi))

class Section:
    """
    Sections must implement all methods to be optimizable
    """
    linear_area: float
    diaphragm_area: float
    i: float
    y_bar: float
    y_top: float
    y_bot: float
    q_neutral: float
    q_glue: List[float]
    glue_widths: List[float]
    one_restrained_sections: Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]
    two_restrained_sections: Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]
    web_restrained_sections: Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]
    web_shear_sections: List[Tuple[float, float]]
    rectangle_lengths: List[Tuple[float, int]]
    diaphragm_rectangle: List[Tuple[float, float]]

    def update(self):
        self.linear_area = self.get_linear_area()
        self.diaphragm_area = self.get_diaphragm_area()
        self.y_bar = self.get_y_bar()
        self.y_top = self.get_y_top()
        self.y_bot = self.get_y_bot()
        self.i = self.get_i()
        self.q_neutral = self.get_q_neutral()
        self.q_glue = self.get_q_glue()
        self.glue_widths = self.get_glue_widths()
        self.one_restrained_sections = self.get_one_restrained_sections()
        self.two_restrained_sections = self.get_two_restrained_sections()
        self.web_restrained_sections = self.get_web_restrained_sections()
        self.web_shear_sections = self.get_web_shear_sections()
        self.rectangle_lengths = self.get_rectangle_lengths()
        self.diaphragm_rectangle = self.get_diaphragm_rectangle()

    def get_linear_area(self) -> float:
        """
        Gets the linear area in mm^2 / mm
        """
        raise NotImplementedError("Sections must have a linear area")

    def get_diaphragm_area(self) -> float:
        """
        Gets the area of one diaphragm in mm^2
        """
        raise NotImplementedError("Sections must have a diaphragm area")

    def get_i(self) -> float:
        """
        Gets the second moment of area of the section
        """
        raise NotImplementedError("Sections must have an i value")

    def get_y_bar(self) -> float:
        """
        Gets the position of the neutral axis
        """
        raise NotImplementedError("Sections must have a y_bar value")

    def get_y_top(self) -> float:
        """
        Gets the distance from y_bar to the top
        """
        raise NotImplementedError("Sections must have a y_top value")

    def get_y_bot(self) -> float:
        """
        Gets the distance from y_bar to the bottom
        """
        raise NotImplementedError("Sections must have a y_bot value")

    def get_q_neutral(self) -> float:
        """
        Gets the q value around the neutral axis
        """
        raise NotImplementedError("Sections must have a q_neutral value")

    def get_glue_widths(self) -> List[float]:
        """
        Gets the widths of the glue sections
        """
        raise NotImplementedError("Section must have a glue_width value")

    def get_q_glue(self) -> List[float]:
        """
        Gets the q values for glue sections
        """
        raise NotImplementedError("Sections must have a q_glue value")

    def get_one_restrained_sections(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Returns a list of pairs, (thickness, width), defining the one-restrained sections in compression above and below the neutral axis
        """
        raise NotImplementedError("Sections must supply one restrained section measurements")

    def get_two_restrained_sections(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Returns a list of pairs, (thickness, width), defining the two-restrained sections in compression
        """
        raise NotImplementedError("Sections must supply two restrained section measurements")

    def get_web_restrained_sections(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Returns a list of pairs, (thickness, height), defining the web sections in compression
        """
        raise NotImplementedError("Sections must supply web restrained section measurements")

    def get_web_shear_sections(self) -> List[Tuple[float, float]]:
        """
        Returns a list of pairs, (thickness, height), defining the web sections
        """
        raise NotImplementedError("Sections must supply web shear section measurements")

    def get_rectangle_lengths(self) -> List[Tuple[float, int]]:
        """
        Returns a list of rectangle heights that can be multiplied by the bridge length to get the rectanlge area
        """
        raise NotImplementedError("Sections must supply rectangle lengths")

    def get_diaphragm_rectangle(self) -> List[Tuple[float, float]]:
        """
        Returns the widthxheight of the diaphragms
        """
        raise NotImplementedError("Sections must supply the diaphragm rectangle")

class Bridge:
    """
    Allows the optimizer to find the correct moment and shear for points or sections
    """

    def __init__(self, forces: List[Tuple[float, float]], reaction_positions: List[float], length: float, a: float, force_diaphragm_space: float = 10):
        """
        forces defines the (position, magnitude) of applied loads starting at position = 0
        reaction_positions defines the points at which there is a reaction force applied starting at 0
        length defines the total length of the bridge
        """
        self.force_diaphragm_space = force_diaphragm_space
        self.length = length
        self.applied_forces = forces
        self.reaction_positions = reaction_positions
        self.reactions = self.get_reactions(reaction_positions)
        self.a = a
        self.loads = sorted(self.applied_forces + self.reactions, key=lambda elem: elem[0])
        if self.loads[0][0] != 0:
            self.loads.insert(0, (0, 0))
        if self.loads[-1][0] != length:
            self.loads.append((length, 0))
        self.diaphragms = self.get_diaphragm_positions()
        self.sections = self.get_sections()
        self.critical_points = self.get_critical_points()
        self.critical_positions = np.array([point[0] for point in self.critical_points])

    def add_force(self, force: Tuple[float, float]):
        """
        Adds a new force to loads and updates the diaphragms
        """
        self.applied_forces += [force]
        self.reactions = self.get_reactions(self.reaction_positions)
        self.loads = sorted(self.applied_forces + self.reactions, key=lambda elem: elem[0])
        if self.loads[0][0] != 0:
            self.loads.insert(0, (0, 0))
        if self.loads[-1][0] != self.length:
            self.loads.append((self.length, 0))
        self.update_sections()

    def update_sections(self, a: float = None):
        if a is not None:
            self.a = a
        self.diaphragms = self.get_diaphragm_positions()
        self.sections = self.get_sections()
        self.critical_points = self.get_critical_points()
        self.critical_positions = np.array([point[0] for point in self.critical_points])

    def num_diaphragms(self):
        return len(self.diaphragms)

    def get_diaphragm_positions(self) -> List[float]:
        diaphragms = set()
        num_extra_diaphragms = int(np.ceil(self.length / self.a))
        edge_space = (self.length - self.a * (num_extra_diaphragms - 1)) / 2
        for position, force in self.loads:
            if abs(force) > 0:
                d_pos_1 = squeeze(position - self.force_diaphragm_space / 2, 0, self.length)
                d_pos_2 = squeeze(position + self.force_diaphragm_space / 2, 0, self.length)
                diaphragms.add(d_pos_1)
                diaphragms.add(d_pos_2)
        for i in range(num_extra_diaphragms):
            if i == 0:
                new_position = edge_space
                # diaphragms.add(edge_space)
            elif i == num_extra_diaphragms:
                new_position = self.length - edge_space
                # diaphragms.add(self.length - edge_space)
            else:
                new_position = edge_space + self.a * i
                # diaphragms.add(edge_space + self.a * i)
            distances = np.abs(np.array(list(diaphragms)) - new_position)
            if np.min(distances) > 2*self.force_diaphragm_space:
                # If the diaphragm is going to be right next to an existing one, might as well save the area and not add it
                diaphragms.add(new_position)
        return sorted(list(diaphragms))
            
    def get_sections(self) -> List[Tuple[float, float]]:
        """
        Defines section (start, end) values based on the bridge length and a value
        """
        sections = []
        for i in range(len(self.diaphragms) - 1):
            start = self.diaphragms[i]
            end = self.diaphragms[i + 1]
            sections.append((start, end))
        return sections

    def get_reactions(self, positions):
        """
        Gets the magnitudes of reaction forces by solving moment and y statics equations
        """
        r_one_pos, r_two_pos = positions
        total_moment = sum([load * (r_one_pos - pos) for pos, load in self.applied_forces])
        r_two_force = total_moment / (r_two_pos - r_one_pos)
        r_one_force = -1 * (sum([force[1] for force in self.applied_forces]) + r_two_force)
        return [(r_one_pos, r_one_force), (r_two_pos, r_two_force)]

    def support_section(self):
        """
        Creates a new force in the center of the bridge 
        """
        r = 1.6629 / 2
        # print(f"Adding a support force. NOTE: This only works for the project force configuration with one section type. To get the support force, multiply the applied force by {r}")
        self.add_force((self.length / 2, r))

    def get_critical_points(self):
        critical_points = []
        curr_shear = 0
        curr_moment = 0
        for i in range(0, len(self.loads)):
            curr_pos, curr_force = self.loads[i]
            curr_shear += curr_force
            if i == 0:
                critical_points.append((curr_pos, curr_shear, curr_moment))
                continue
            last_pos, last_force = self.loads[i-1]
            curr_moment += (curr_pos - last_pos) * (curr_shear - curr_force)
            critical_points.append((curr_pos, curr_shear, curr_moment))
        return critical_points

    def get_adjacent_critical_points(self, pos):
        """
        Gets the ciritical points just below and just above the given position
        """
        if pos >= self.length or pos < 0:
            raise ValueError('Position must be between [0 and bridge length)')
        valid_indices = np.where(np.array(self.critical_positions) > pos)[0]
        next_index = valid_indices[0]
        return [self.critical_points[next_index - 1], self.critical_points[next_index]]

    def get_shear_moment(self, pos):
        """
        Gets the (shear, moment) at the specified position
        """
        if pos in self.critical_positions:
            index = np.where(self.critical_positions == pos)[0][0]
            return self.critical_points[index][1:]
        last_crit, next_crit = self.get_adjacent_critical_points(pos)
        shear = last_crit[1]
        moment = ((pos - last_crit[0])) * ((next_crit[2] - last_crit[2]) / (next_crit[0] - last_crit[0])) + last_crit[2]
        return abs(shear), moment

    def get_max_shear_moment(self, start, end):
        start_shear, start_moment = self.get_shear_moment(start)
        end_shear, end_moment = self.get_shear_moment(end)
        valid_indices = np.where(np.logical_and(self.critical_positions < end, self.critical_positions > start))[0]

        possible_moments = [start_moment, end_moment] + [self.critical_points[i][2] for i in valid_indices]
        possible_shears = [start_shear, end_shear] + [abs(self.critical_points[i][1]) for i in valid_indices]

        max_positive_moment = max(0, *possible_moments)
        max_negative_moment = min(0, *possible_moments)

        return max(possible_shears), max_positive_moment, abs(max_negative_moment)

class Material:
    tensile_strength: float
    compressive_strength: float
    shear_strength: float
    glue_shear_strength: float
    stiffness: float
    poisson: float

    def __init__(self, tensile_strength = 30, compressive_strength = 6, shear_strength = 4, glue_shear_strength = 2, stiffness = 40000, poisson = 0.2):
        self.tensile_strength = tensile_strength
        self.compressive_strength = compressive_strength
        self.shear_strength = shear_strength
        self.glue_shear_strength = glue_shear_strength
        self.stiffness = stiffness
        self.poisson = poisson

class MaxForceSolver:
    def __init__(self, section: Section, material: Material, bridge: Bridge):
        self.section = section
        self.material = material
        self.bridge = bridge
        self.failure_modes = ["Tensile", "Crushing", "Section Shear", "Glue Shear", "Two Restrained Buckling", "One Restrained Buckling", "Web Buckling", "Web Shear"]
        self.failure_equations = [self.tensile_failure_load, self.crushing_failure_load, self.shear_failure_load, self.glue_failure_load, self.two_restrained_failure_load, self.one_restrained_failure_load, self.web_buckling_failure_load, self.web_shear_failure_load]

    def set_section(self, section: Section):
        self.section = section

    def set_material(self, material: Material):
        self.material = material
    
    def set_bridge(self, bridge: Bridge):
        self.bridge = bridge

    def section_tensile_failure_load(self, start, end):
        shear, positive_moment, negative_moment = self.bridge.get_max_shear_moment(start, end)
        positive_failure = (self.material.tensile_strength * self.section.i) / (positive_moment * self.section.y_bot) if positive_moment > 0 else np.inf
        negative_failure = (self.material.tensile_strength * self.section.i) / (negative_moment * self.section.y_top) if negative_moment > 0 else np.inf
        return min(positive_failure, negative_failure)

    def tensile_failure_load(self):
        loads = [self.section_tensile_failure_load(*sec) for sec in self.bridge.sections]
        index = np.argmin(loads)
        return loads[index], index

    def section_crushing_failure_load(self, start, end):
        shear, positive_moment, negative_moment = self.bridge.get_max_shear_moment(start, end)
        positive_failure = (self.material.compressive_strength * self.section.i) / (positive_moment * self.section.y_top) if positive_moment > 0 else np.inf
        negative_failure = (self.material.compressive_strength * self.section.i) / (negative_moment * self.section.y_bot) if negative_moment > 0 else np.inf
        return min(positive_failure, negative_failure)
    
    def crushing_failure_load(self):
        loads = [self.section_crushing_failure_load(*sec) for sec in self.bridge.sections]
        index = np.argmin(loads)
        return loads[index], index

    def section_shear_failure_load(self, start, end):
        shear, *moments = self.bridge.get_max_shear_moment(start, end)
        web_thickness = self.section.get_web_shear_sections()[0][0]
        if shear == 0:
            return np.inf
        max_force = (2 * self.material.shear_strength * web_thickness * self.section.i) / (shear * self.section.q_neutral)
        return max_force

    def shear_failure_load(self):
        loads = [self.section_shear_failure_load(*sec) for sec in self.bridge.sections]
        index = np.argmin(loads)
        return loads[index], index

    def section_glue_failure_load(self, start, end):
        shear, *moments = self.bridge.get_max_shear_moment(start, end)
        if shear == 0:
            return np.inf
        failure_loads = []
        num_glue_sections = len(self.section.glue_widths)
        for i in range(num_glue_sections):
            glue_width = self.section.glue_widths[i]
            q_glue = self.section.q_glue[i]
            failure_loads.append((self.material.glue_shear_strength * glue_width * self.section.i) / (shear * q_glue))
        return min(failure_loads)

    def glue_failure_load(self):
        loads = [self.section_glue_failure_load(*sec) for sec in self.bridge.sections]
        index = np.argmin(loads)
        return loads[index], index

    def section_two_restrained_load(self, start, end):
        shear, positive_moment, negative_moment = self.bridge.get_max_shear_moment(start, end)
        positive_failure = 0
        positive_sections = self.section.two_restrained_sections[0]
        if positive_moment == 0 or len(positive_sections) == 0:
            positive_failure = np.inf
        else:
            failure_loads = []
            for thickness, width in positive_sections:
                critical_stress = ((4 * np.pi**2 * self.material.stiffness) / (12 * (1 - self.material.poisson**2))) * (thickness / width)**2
                failure_loads.append((critical_stress * self.section.i) / (positive_moment * self.section.y_top))
            positive_failure = min(failure_loads)

        negative_failure = 0
        negative_sections = self.section.two_restrained_sections[1]
        if negative_moment == 0 or len(negative_sections) == 0:
            negative_failure = np.inf
        else:
            failure_loads = []
            for thickness, width in negative_sections:
                critical_stress = ((4 * np.pi**2 * self.material.stiffness) / (12 * (1 - self.material.poisson**2))) * (thickness / width)**2
                failure_loads.append((critical_stress * self.section.i) / (negative_moment * self.section.y_bot))
            negative_failure = min(failure_loads)
        return min(positive_failure, negative_failure)

    def two_restrained_failure_load(self):
        loads = [self.section_two_restrained_load(*sec) for sec in self.bridge.sections]
        index = np.argmin(loads)
        return loads[index], index

    def section_one_restrained_load(self, start, end):
        shear, positive_moment, negative_moment = self.bridge.get_max_shear_moment(start, end)

        positive_failure = 0
        positive_sections = self.section.one_restrained_sections[0]
        if positive_moment == 0 or len(positive_sections) == 0:
            positive_failure = np.inf
        else:
            failure_loads = []
            for thickness, width in positive_sections:
                critical_stress = ((0.425 * np.pi**2 * self.material.stiffness) / (12 * (1 - self.material.poisson**2))) * (thickness / width)**2
                failure_loads.append((critical_stress * self.section.i) / (positive_moment * self.section.y_top))
            positive_failure = min(failure_loads)

        negative_failure = 0
        negative_sections = self.section.one_restrained_sections[1]
        if negative_moment == 0 or len(negative_sections) == 0:
            negative_failure = np.inf
        else:
            failure_loads = []
            for thickness, width in negative_sections:
                critical_stress = ((0.425 * np.pi**2 * self.material.stiffness) / (12 * (1 - self.material.poisson**2))) * (thickness / width)**2
                failure_loads.append((critical_stress * self.section.i) / (negative_moment * self.section.y_bot))
            negative_failure = min(failure_loads)

        return min(positive_failure, negative_failure)

    def one_restrained_failure_load(self):
        loads = [self.section_one_restrained_load(*sec) for sec in self.bridge.sections]
        index = np.argmin(loads)
        return loads[index], index

    def section_web_buckling_load(self, start, end):
        shear, positive_moment, negative_moment = self.bridge.get_max_shear_moment(start, end)

        positive_failure = 0
        positive_sections = self.section.web_restrained_sections[0]
        if positive_moment == 0 or len(positive_sections) == 0:
            positive_failure = np.inf
        else:
            failure_loads = []
            for thickness, height in positive_sections:
                critical_stress = ((6 * np.pi**2 * self.material.stiffness) / (12 * (1 - self.material.poisson**2))) * (thickness / height)**2
                failure_loads.append((critical_stress * self.section.i) / (positive_moment * height))
            positive_failure = min(failure_loads)

        negative_failure = 0
        negative_sections = self.section.web_restrained_sections[1]
        if negative_moment == 0 or len(negative_sections) == 0:
            negative_failure = np.inf
        else:
            failure_loads = []
            for thickness, height in negative_sections:
                critical_stress = ((6 * np.pi**2 * self.material.stiffness) / (12 * (1 - self.material.poisson**2))) * (thickness / height)**2
                failure_loads.append((critical_stress * self.section.i) / (negative_moment * height))
            negative_failure = min(failure_loads)

        return min(positive_failure, negative_failure)

    def web_buckling_failure_load(self):
        loads = [self.section_web_buckling_load(*sec) for sec in self.bridge.sections]
        index = np.argmin(loads)
        return loads[index], index

    def section_web_shear_load(self, start, end):
        shear, *moment = self.bridge.get_max_shear_moment(start, end)
        a = abs(end - start)
        if shear == 0:
            return np.inf
        failure_loads = []
        for thickness, height in self.section.web_shear_sections:
            critical_shear = ((5 * np.pi**2 * self.material.stiffness) / (12 * (1 - self.material.poisson**2))) * ((thickness / height)**2 + (thickness / a)**2)
            failure_loads.append((2 * critical_shear * self.section.i * thickness) / (shear * self.section.q_neutral))
        return min(failure_loads)

    def web_shear_failure_load(self):
        loads = [self.section_web_shear_load(*sec) for sec in self.bridge.sections]
        index = np.argmin(loads)
        return loads[index], index

    def area(self):
        length = self.bridge.length
        linear_area = self.section.linear_area
        diaphragm_area = self.section.diaphragm_area
        beam_area = linear_area * length
        diaphragm_area = self.bridge.num_diaphragms() * diaphragm_area
        return beam_area + diaphragm_area

    def rectangles(self):
        rectangles = []
        num_diaphragms = self.bridge.num_diaphragms()
        diaphragm_dims = self.section.diaphragm_rectangle

        bridge_rect_lengths = self.section.rectangle_lengths

        for rect_length in bridge_rect_lengths:
            rectangles.extend([(rect_length[0], self.bridge.length) for i in range(rect_length[1])])
        rectangles.extend([diaphragm_dims for i in range(num_diaphragms)])

        return rectangles

    def failure_load(self):
        failure_loads = [load_equation() for load_equation in self.failure_equations]
        failure_index = np.argmin([load[0] for load in failure_loads])
        return failure_loads[failure_index][0], self.failure_modes[failure_index], failure_loads[failure_index][1], self.bridge.sections[failure_loads[failure_index][1]]

class PiSection(Section):
    def __init__(self, width: float, web_height: float, flange_width: float, glue_width: float, center_deck_sheets: int, flange_sheets: int, thickness: float = 1.27):
        super().__init__()
        self.width = width
        self.web_height = web_height
        self.flange_width = flange_width
        self.glue_width = glue_width
        self.deck_sheets = center_deck_sheets
        self.flange_sheets = flange_sheets
        self.thickness = thickness
        self.update()

    def get_linear_area(self) -> float:
        """
        Gets the linear area in mm^2 / mm
        """
        webbing = self.web_height + self.glue_width - self.thickness
        deck = 2*self.flange_width*self.flange_sheets + (self.width - 2*self.flange_width)*self.deck_sheets
        return 2*webbing + deck

    def get_diaphragm_area(self) -> float:
        """
        Gets the area of one diaphragm in mm^2
        """
        return self.web_height * (self.width - 2*self.flange_width - 2*self.thickness)

    def get_y_bar(self) -> float:
        """
        Gets the position of the neutral axis
        """
        flange_area = 2*self.thickness*self.flange_sheets*self.flange_width
        flange_distance = self.web_height + (self.thickness * self.flange_sheets) / 2

        deck_area = self.thickness * self.deck_sheets * (self.width - 2*self.flange_width)
        deck_distance = self.web_height + (self.thickness * self.deck_sheets) / 2

        web_area = self.web_height * self.thickness * 2
        web_distance = self.web_height / 2

        glue_area = 2 * self.thickness * (self.glue_width - self.thickness)
        glue_distance = self.web_height - (self.thickness / 2)

        return (flange_area*flange_distance + deck_area*deck_distance + web_area*web_distance + glue_area*glue_distance) / (flange_area + deck_area + web_area + glue_area)

    def get_y_top(self) -> float:
        """
        Gets the distance from y_bar to the top
        """
        sheets = max(self.deck_sheets, self.flange_sheets)
        return self.web_height + self.thickness * sheets - self.y_bar

    def get_y_bot(self) -> float:
        """
        Gets the distance from y_bar to the bottom
        """
        return self.y_bar

    def get_i(self) -> float:
        """
        Gets the second moment of area of the section
        """
        flange_area = self.thickness*self.flange_sheets*self.flange_width
        flange_i = (self.flange_width * (self.thickness*self.flange_sheets)**3) / 12
        flange_d = self.web_height + (self.thickness*self.flange_sheets) / 2 - self.y_bar

        deck_area = self.thickness * self.deck_sheets * (self.width - 2*self.flange_width)
        deck_i = ((self.width - 2*self.flange_width) * (self.thickness*self.flange_sheets)**3) / 12
        deck_d = self.web_height + (self.thickness * self.deck_sheets) / 2 - self.y_bar

        web_area = self.web_height * self.thickness
        web_i = (self.thickness * self.web_height**3) / 12
        web_d = self.web_height / 2 - self.y_bar

        glue_area = self.thickness * (self.glue_width - self.thickness)
        glue_i = ((self.glue_width - self.thickness) * self.thickness**3) / 12
        glue_d = self.web_height - (self.thickness / 2) - self.y_bar

        flange_I = 2*(flange_i + flange_area*flange_d**2)
        deck_I = deck_i + deck_area*deck_d**2
        web_I = 2*(web_i + web_area*web_d**2)
        glue_I = 2*(glue_i + glue_area*glue_d**2)

        return flange_I + deck_I + web_I + glue_I

    def get_q_neutral(self) -> float:
        """
        Gets the q value around the neutral axis
        """
        # We take q around the bottom of the webbing
        area = self.y_bar * self.thickness
        d = self.y_bar / 2
        return 2 * area * d

    def get_glue_widths(self) -> List[float]:
        """
        Gets the widths of the glue sections
        """
        return [self.glue_width * 2]

    def get_q_glue(self) -> List[float]:
        """
        Gets the q values for glue sections
        """
        # We take q above the glue
        flange_area = self.thickness*self.flange_sheets*self.flange_width * 2
        flange_d = self.web_height + (self.thickness*self.flange_sheets) / 2 - self.y_bar

        deck_area = self.thickness * self.deck_sheets * (self.width - 2*self.flange_width)
        deck_d = self.web_height + (self.thickness * self.deck_sheets) / 2 - self.y_bar

        return [flange_area*flange_d + deck_area*deck_d]


    def get_one_restrained_sections(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Returns a list of pairs, (thickness, width), defining the one-restrained sections in compression
        """
        return ([(self.thickness*self.flange_sheets, self.flange_width)], [])

    def get_two_restrained_sections(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Returns a list of pairs, (thickness, width), defining the two-restrained sections in compression
        """
        return ([(self.thickness*self.deck_sheets, self.width - 2*self.flange_width)], [])

    def get_web_restrained_sections(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Returns a list of pairs, (thickness, height), defining the web sections in compression
        """
        return ([(self.thickness, self.web_height - self.y_bar)], [(self.thickness, self.y_bar)])

    def get_web_shear_sections(self) -> List[Tuple[float, float]]:
        """
        Returns a list of pairs, (thickness, height), defining the web sections
        """
        return [(self.thickness, self.web_height)]

    def get_rectangle_lengths(self) -> List[Tuple[float, int]]:
        """
        Returns a list of lengths that when multiplied by the bridge length give the area.
        """
        rects = []
        # Deck
        rects.append((float2dec(self.width, 2), 2))
        # Webs
        rects.append((float2dec(self.web_height + (self.glue_width - 1.27), 2), 2))
        return rects

    def get_diaphragm_rectangle(self) -> List[Tuple[float, float]]:
        """
        Return the widthxheight of the diaphragms.
        """
        return [float2dec(self.width - 2*self.flange_width - 2*self.thickness, 3), float2dec(self.web_height, 3)]

class ISection(Section):
    def __init__(self, top_width: float, top_sheets: int, top_glue_width: float, bottom_width: float, bottom_sheets: int, bottom_glue_width: float, web_width: float, web_height: float, thickness: float = 1.27):
        super().__init__()
        self.top_width = top_width
        self.top_sheets = top_sheets
        self.top_glue_width = top_glue_width
        self.bottom_width = bottom_width
        self.bottom_sheets = bottom_sheets
        self.bottom_glue_width = bottom_glue_width
        self.web_width = web_width
        self.web_height = web_height
        self.thickness = thickness
        self.update()

    def get_linear_area(self) -> float:
        top_deck = self.top_sheets * self.top_width
        bottom_deck = self.bottom_sheets * self.bottom_width
        top_glue = self.top_glue_width - self.thickness
        bottom_glue = self.bottom_glue_width - self.thickness
        web = self.web_height
        return top_deck + bottom_deck + 2*(top_glue + bottom_glue + web)

    def get_diaphragm_area(self) -> float:
        return self.web_width * self.web_height

    def get_y_bar(self) -> float:
        top_deck_a = self.top_sheets * self.top_width * self.thickness
        top_deck_h = self.bottom_sheets * self.thickness + self.web_height + (self.top_sheets*self.thickness) / 2

        bottom_glue_a = (self.bottom_glue_width - self.thickness) * self.thickness
        bottom_glue_h = self.bottom_sheets * self.thickness + self.thickness / 2

        web_a = self.web_height * self.thickness
        web_h = self.bottom_sheets * self.thickness + self.web_height / 2

        top_glue_a = (self.top_glue_width - self.thickness) * self.thickness
        top_glue_h = self.bottom_sheets * self.thickness + self.web_height - self.thickness / 2

        bottom_deck_a = self.bottom_sheets * self.bottom_width * self.thickness
        bottom_deck_h = (self.bottom_sheets * self.thickness) / 2

        return (top_deck_a*top_deck_h + 2*bottom_glue_a*bottom_glue_h + 2*web_a*web_h + 2*top_glue_a*top_glue_h + bottom_deck_a*bottom_deck_h) / (top_deck_a + 2*bottom_glue_a + 2*web_a + 2*top_glue_a + bottom_deck_a)

    def get_y_top(self) -> float:
        return self.bottom_sheets*self.thickness + self.web_height + self.top_sheets*self.thickness - self.y_bar

    def get_y_bot(self) -> float:
        return self.y_bar

    def get_i(self) -> float:
        bottom_deck_i = (self.bottom_width * (self.bottom_sheets * self.thickness)**3) / 12
        bottom_deck_d = (self.bottom_sheets * self.thickness) / 2 - self.y_bar
        bottom_deck_a = self.bottom_sheets * self.bottom_width * self.thickness
        bottom_deck_I = bottom_deck_i + bottom_deck_a * bottom_deck_d**2

        bottom_glue_i = ((self.bottom_glue_width - self.thickness) * self.thickness**3) / 12
        bottom_glue_d = self.bottom_sheets * self.thickness + self.thickness / 2 - self.y_bar
        bottom_glue_a = (self.bottom_glue_width - self.thickness) * self.thickness
        bottom_glue_I = bottom_deck_i + bottom_glue_a * bottom_glue_d**2

        web_i = (self.thickness * self.web_height**3) / 12
        web_d = self.bottom_sheets * self.thickness + self.web_height / 2 - self.y_bar
        web_a = self.web_height * self.thickness
        web_I = web_i + web_a * web_d**2

        top_glue_i = ((self.top_glue_width - self.thickness) * self.thickness**3) / 12
        top_glue_d = self.bottom_sheets * self.thickness + self.web_height - self.thickness / 2 - self.y_bar
        top_glue_a = (self.top_glue_width - self.thickness) * self.thickness
        top_glue_I = top_glue_i + top_glue_a * top_glue_d**2

        top_deck_i = (self.top_width * (self.top_sheets * self.thickness)**3) / 12
        top_deck_d = self.bottom_sheets * self.thickness + self.web_height + (self.top_sheets*self.thickness) / 2 - self.y_bar
        top_deck_a = self.top_sheets * self.top_width * self.thickness
        top_deck_I = top_deck_i + top_deck_a * top_deck_d**2

        return bottom_deck_I + 2*(bottom_glue_I + web_I + top_glue_I) + top_deck_I

    def get_q_neutral(self) -> float:
        web_a = (self.y_bar - self.bottom_sheets * self.thickness) * self.thickness
        web_d = self.y_bar - (self.bottom_sheets * self.thickness + (self.y_bar - self.bottom_sheets * self.thickness) / 2)
        web_q = web_a * abs(web_d)

        glue_a = (self.bottom_glue_width - self.thickness) * self.thickness
        glue_d = self.y_bar - (self.bottom_sheets * self.thickness + self.thickness / 2)
        glue_q = glue_a * abs(glue_d)

        deck_a = self.bottom_sheets * self.bottom_width * self.thickness
        deck_d = self.y_bar - (self.bottom_sheets * self.thickness / 2)
        deck_q = deck_a * abs(deck_d)

        return deck_q + 2*(web_q + glue_q)

    def get_glue_widths(self) -> List[float]:
        """
        Gets the widths of the glue sections
        """
        return [2*self.top_glue_width, 2*self.bottom_glue_width]

    def get_q_glue(self) -> List[float]:
        bottom_deck_a = self.bottom_sheets * self.bottom_width * self.thickness
        bottom_deck_d = self.y_bar - (self.bottom_sheets * self.thickness / 2)
        bottom_deck_q = bottom_deck_a * abs(bottom_deck_d)

        top_deck_a = self.top_sheets * self.top_width * self.thickness
        top_deck_d = self.y_bar - (self.bottom_sheets * self.thickness + self.web_height + (self.top_sheets * self.thickness) / 2)
        top_deck_q = top_deck_a * abs(top_deck_d)

        return [top_deck_q, bottom_deck_q]

    def get_one_restrained_sections(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Gets all (thickness, width) for one restrained sections (above, below) the neutral axis
        """
        top_flange_width = (self.top_width - self.web_width) / 2
        top_flange_thickness = self.top_sheets * self.thickness
        top_flange_desc = (top_flange_thickness, top_flange_width)

        bottom_flange_width = (self.bottom_width - self.web_width) / 2
        bottom_flange_thickness = self.bottom_sheets * self.thickness
        bottom_flange_desc = (bottom_flange_thickness, bottom_flange_width)

        return ([top_flange_desc], [bottom_flange_desc])

    def get_two_restrained_sections(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Gets all (thickness, width) for two restrained sections (above, below) the neutral axis
        """
        top_center_width = self.web_width
        top_center_thickness = self.top_sheets * self.thickness
        top_center_desc = (top_center_thickness, top_center_width)

        bottom_center_width = self.web_width
        bottom_center_thickenss = self.bottom_sheets * self.thickness
        bottom_center_desc = (bottom_center_thickenss, bottom_center_width)

        return ([top_center_desc], [bottom_center_desc])

    def get_web_restrained_sections(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Returns a list of pairs, (thickness, height), defining the web sections in compression for (above, below) the neutral axis
        """
        top_height = self.bottom_sheets * self.thickness + self.web_height - self.y_bar
        top_thickness = self.thickness
        top_desc = (top_thickness, top_height)

        bottom_height = self.y_bar - self.bottom_sheets * self.thickness
        bottom_thickness = self.thickness
        bottom_desc = (bottom_thickness, bottom_height)
        return ([top_desc], [bottom_desc])

    def get_web_shear_sections(self) -> List[Tuple[float, float]]:
        """
        Returns a list of pairs, (thickness, height), defining the web sections
        """
        web_thickness = self.thickness
        web_height = self.web_height
        return [(web_thickness, web_height)]

    def get_rectangle_lengths(self) -> List[Tuple[float, int]]:
        """
        Returns a list of rectangle heights that can be multiplied by the bridge length to get the rectanlge area
        """
        rects = []
        # Top Deck
        rects.append((float2dec(self.top_width, 2), self.top_sheets))
        # Bottom Deck
        rects.append((float2dec(self.bottom_width, 2), self.bottom_sheets))
        # Webbing
        rects.append((float2dec(self.web_height + (self.top_glue_width - 1.27) + (self.bottom_glue_width - 1.27), 2), 2))
        return rects

    def get_diaphragm_rectangle(self) -> List[Tuple[float, float]]:
        """
        Returns the widthxheight of the diaphragms
        """
        return [float2dec(self.web_width, 2), float2dec(self.web_height, 2)]


def pack_rectangles(solver: MaxForceSolver, mat_dims = (813, 1016), outfile = None, extra_rectangles = []) -> Tuple:
    colors = ['r', 'silver', 'tan', 'orange', 'g', 'c', 'm', 'y', 'k', 'w']
    color_map = {}
    rectangles = solver.rectangles()
    extra_rectangles = [(float2dec(r[0], 2), float2dec(r[1], 2)) for r in extra_rectangles]
    rectangles.extend(extra_rectangles)
    packer = newPacker()
    for r in rectangles:
        rect_name = "x".join([str(s) for s in sorted(r[:2])])
        if rect_name not in color_map:
            color_map[rect_name] = colors.pop(0)
        if len(r) == 2:
            packer.add_rect(*r)
        else:
            packer.add_rect(*r[:-1])
    packer.add_bin(*mat_dims)
    try:
        packer.pack()
    except AssertionError:
        print("Invalid Rectangles")
        print(rectangles)
        return 0, len(rectangles)
    nrects = len(packer[0])
    if outfile:
        if len(packer[0]) < len(rectangles):
            print(f"PACKING FAILED: NOT ENOUGH SPACE. {len(packer[0])} out of {len(rectangles)} could fit")
        fig, ax = plt.subplots()
        plt.axis('scaled')
        for i, rect in enumerate(packer[0]):
            rect_name = "x".join([str(s) for s in sorted([rect.width, rect.height])])
            color = color_map[rect_name]
            ax.add_patch(
                patches.Rectangle(
                    (rect.x, rect.y),
                    rect.width,
                    rect.height,
                    edgecolor = 'black',
                    facecolor = color,
                    fill=True
                ))
        ax.set_ylim(bottom=0, top=1016)
        ax.set_xlim(left=0, right=813)
        os.makedirs('./tmp', exist_ok=True)
        plt.savefig(f'./tmp/{outfile}.png', dpi=500)
    packing_prop = nrects / len(rectangles)
    return packing_prop, nrects - len(rectangles)

def optimize_i(top_width_guess, top_glue_width_guess, bottom_width_guess, bottom_glue_width_guess, web_width_guess, web_height_guess, a_guess, top_sheets = 1, bottom_sheets = 2, mat_width = 813, mat_height = 1016, thickness = 1.27, supported = True):
    max_web_height = 200 - top_sheets * thickness - bottom_sheets * thickness
    min_web_height = 10
    max_web_width = 200
    min_web_width = 20
    max_glue_width = 30
    min_glue_width = 5*thickness
    max_width = 300
    min_width = 20
    min_top_width = 100
    min_a = 20
    max_a = 200
    max_area = mat_width * mat_height
    # area_to_leave = max_area * 0.28
    area_to_leave = max_area * 0.31

    def get_force_solver(top_width, top_glue_width, bottom_width, bottom_glue_width, web_width, web_height, a):
        section = ISection(top_width, top_sheets, top_glue_width, bottom_width, bottom_sheets, bottom_glue_width, web_width, web_height, thickness=thickness)
        bridge = Bridge([(280 + 15, -0.5), (280 + 390 + 15, -0.5)], [15, 980-15], 980, a)
        if supported:
            bridge.support_section()
        material = Material(30, 6, 4, 2, 4000, 0.2)
        return MaxForceSolver(section, material, bridge)

    def max_web_height_con(x):
        top_width, top_glue_width, bottom_width, bottom_glue_width, web_width, web_height, a = x
        return max_web_height - web_height
    def min_web_height_con(x):
        top_width, top_glue_width, bottom_width, bottom_glue_width, web_width, web_height, a = x
        return web_height - min_web_height
    def max_web_width_con(x):
        top_width, top_glue_width, bottom_width, bottom_glue_width, web_width, web_height, a = x
        return max_web_width - web_width
    def mix_web_width_con(x):
        top_width, top_glue_width, bottom_width, bottom_glue_width, web_width, web_height, a = x
        return web_width - min_web_width
    def max_top_glue_width_con(x):
        top_width, top_glue_width, bottom_width, bottom_glue_width, web_width, web_height, a = x
        return max_glue_width - top_glue_width
    def max_bottom_glue_width_con(x):
        top_width, top_glue_width, bottom_width, bottom_glue_width, web_width, web_height, a = x
        return max_glue_width - bottom_glue_width
    def min_top_glue_width_con(x):
        top_width, top_glue_width, bottom_width, bottom_glue_width, web_width, web_height, a = x
        return top_glue_width - min_glue_width
    def min_bottom_glue_width_con(x):
        top_width, top_glue_width, bottom_width, bottom_glue_width, web_width, web_height, a = x
        return bottom_glue_width - min_glue_width
    def max_top_width_con(x):
        top_width, top_glue_width, bottom_width, bottom_glue_width, web_width, web_height, a = x
        return max_width - top_width
    def max_bottom_width_con(x):
        top_width, top_glue_width, bottom_width, bottom_glue_width, web_width, web_height, a = x
        return max_width - bottom_width
    def min_top_width_con(x):
        top_width, top_glue_width, bottom_width, bottom_glue_width, web_width, web_height, a = x
        return top_width - min_top_width
    def min_top_width_con_2(x):
        top_width, top_glue_width, bottom_width, bottom_glue_width, web_width, web_height, a = x
        return top_width - web_width - 2*top_glue_width
    def min_bottom_width_con(x):
        top_width, top_glue_width, bottom_width, bottom_glue_width, web_width, web_height, a = x
        return bottom_width - min_width
    def min_bottom_width_con_2(x):
        top_width, top_glue_width, bottom_width, bottom_glue_width, web_width, web_height, a = x
        return bottom_width - web_width - 2*bottom_glue_width
    def max_area_con(x):
        top_width, top_glue_width, bottom_width, bottom_glue_width, web_width, web_height, a = x
        solver = get_force_solver(*x)
        return max_area - solver.area() - area_to_leave
    def max_a_con(x):
        top_width, top_glue_width, bottom_width, bottom_glue_width, web_width, web_height, a = x
        return max_a - a
    def min_a_con(x):
        top_width, top_glue_width, bottom_width, bottom_glue_width, web_width, web_height, a = x
        return a - min_a

    def optimize(x):
        top_width, top_glue_width, bottom_width, bottom_glue_width, web_width, web_height, a = x
        if a < 0:
            return np.inf
        solver = get_force_solver(*x)
        best = solver.failure_load()
        # print(best)
        # print("Running optimize", web_height, flange_width, glue_width, get_max_force(web_height, flange_width, glue_width))
        return -1 * best[0]

    cons = [
        {'type': 'ineq', 'fun': max_web_height_con},
        {'type': 'ineq', 'fun': min_web_height_con},
        {'type': 'ineq', 'fun': max_web_width_con},
        {'type': 'ineq', 'fun': mix_web_width_con},
        {'type': 'ineq', 'fun': max_top_glue_width_con},
        {'type': 'ineq', 'fun': max_bottom_glue_width_con},
        {'type': 'ineq', 'fun': min_top_glue_width_con},
        {'type': 'ineq', 'fun': min_bottom_glue_width_con},
        {'type': 'ineq', 'fun': max_top_width_con},
        {'type': 'ineq', 'fun': min_top_width_con},
        {'type': 'ineq', 'fun': max_bottom_width_con},
        {'type': 'ineq', 'fun': max_area_con},
        {'type': 'ineq', 'fun': max_a_con},
        {'type': 'ineq', 'fun': min_a_con},
        {'type': 'ineq', 'fun': min_top_width_con_2},
        {'type': 'ineq', 'fun': min_bottom_width_con_2},
    ]

    res = minimize(optimize, [top_width_guess, top_glue_width_guess, bottom_width_guess, bottom_glue_width_guess, web_width_guess, web_height_guess, a_guess], constraints=cons)
    solver = get_force_solver(*res.x)
    r = list(res.x[:])
    r.append(solver)
    return r

def search_i(params, supported = True):
    top_width_guess, top_glue_width_guess, bottom_width_guess, bottom_glue_width_guess, web_width_guess, web_height_guess, a_guess = params
    *values, solver = optimize_i(top_width_guess, top_glue_width_guess, bottom_width_guess, bottom_glue_width_guess, web_width_guess, web_height_guess, a_guess, supported = supported)
    failure_load, reason, section_num, section = solver.failure_load()
    values.insert(0, failure_load)
    values.append(solver)
    return values

def search_area_i(t_width_mm, b_width_mm, glue_mm, w_width_mm, w_height_mm, a_mm, mat_width = 813, mat_height = 1016, supported = True):
    max_area = mat_width * mat_height
    space_t_w = np.linspace(*t_width_mm)
    space_b_w = np.linspace(*b_width_mm)
    space_g = np.linspace(*glue_mm)
    space_w_w = np.linspace(*w_width_mm)
    space_w_h = np.linspace(*w_height_mm)
    space_a = np.linspace(*a_mm)

    best = None
    count = 0
    total = len(space_t_w) * len(space_b_w) * len(space_g) * len(space_w_w) * len(space_w_h) * len(space_a)
    for t_w in space_t_w:
        for b_w in space_b_w:
            for g in space_g:
                for w_w in space_w_w:
                    for w_h in space_w_h:
                        for a in space_a:
                            supporter_width = 20
                            extra_rects = [(45, 600), (45, 600), (supporter_width, 600), (supporter_width, 600), (supporter_width, 600), (supporter_width, 600), (42.46, 600), (42.46, 600), (42.46, 600), (20.595, 600), (20.595, 600)]
                            failure_load, *params, solver = search_i([t_w, g, b_w, g, w_w, w_h, a], supported=supported)
                            packing_prop, diff = pack_rectangles(solver, extra_rectangles=extra_rects)
                            if diff == 0:
                                if best is None or failure_load > best[0]:
                                    best = [failure_load, *params, solver]
                                    print(f"New Best: {best[0]} -> {failure_load}")
                                else:
                                    print(f"Worse: {failure_load} < {best[0]}")
                            else:
                                print(f"Rectangles did not pack: {failure_load}")
                            count += 1
                            print(f"{count}/{total} ", failure_load, [t_w, g, b_w, g, w_w, w_h, a], params)
    return best
    

def optimize_pi(web_height_guess = 100, flange_guess = 20, glue_guess = 5, a_guess = 100, width_guess: float = 120, deck_sheets=2, flange_sheets=2, max_area = 813 * 1016, mat_width = 813, mat_height = 1016, extra_rectangles: List[Tuple[float, float]] = [], thickness = 1.27, supported = False):
    max_web_height = 200 - deck_sheets * thickness
    max_glue_width = 30
    max_width = 100000
    min_width = 100
    min_center_deck = 20
    area_to_leave = max_area * 0.3

    def get_force_solver(width, web_height, flange_width, glue_width, a):
        section = PiSection(width, web_height, flange_width, glue_width, deck_sheets, flange_sheets, thickness)
        bridge = Bridge([(280 + 15, -0.5), (280 + 390 + 15, -0.5)], [15, 980-15], 980, a)
        if supported:
            bridge.support_section()
        material = Material(30, 6, 4, 2, 4000, 0.2)
        return MaxForceSolver(section, material, bridge)

    def packing_con(x):
        solver = get_force_solver(*x)
        packing_prop, diff = pack_rectangles(solver, (mat_width, mat_height), extra_rectangles=extra_rectangles)
        return diff
    def area_con(x):
        solver = get_force_solver(*x)
        return max_area - solver.area() - area_to_leave
    def flange_con(x):
        width, web_height, flange_width, glue_width, a = x
        return flange_width
    def flange_con_2(x):
        width, web_height, flange_width, glue_width, a = x
        if flange_sheets > deck_sheets:
            return width - 2*flange_width - 100 - min_center_deck / 2
        else:
            return width / 2 - 2 * glue_width - flange_width - min_center_deck / 2
    def width_con(x):
        width, web_height, flange_width, glue_width, a = x
        return max_width - width
    def width_con_2(x):
        width, web_height, flange_width, glue_width, a = x
        return width - 2 * flange_width
    def width_con_3(x):
        width, web_height, flange_width, glue_width, a = x
        return width - min_width
    def web_con(x):
        width, web_height, flange_width, glue_width, a = x
        return web_height
    def web_con_2(x):
        width, web_height, flange_width, glue_width, a = x
        return max_web_height - web_height
    def glue_con(x):
        width, web_height, flange_width, glue_width, a = x
        return glue_width - thickness
    def glue_con_2(x):
        width, web_height, flange_width, glue_width, a = x
        return max_glue_width - glue_width
    def a_con(x):
        width, web_height, flange_width, glue_width, a = x
        return a - 20
    def optimize(x):
        width, web_height, flange_width, glue_width, a = x
        if a < 0:
            return np.inf
        solver = get_force_solver(*x)
        best = solver.failure_load()
        # print(best)
        # print("Running optimize", web_height, flange_width, glue_width, get_max_force(web_height, flange_width, glue_width))
        return -1 * best[0]

    cons = [
        {'type': 'ineq', 'fun': area_con},
        {'type': 'ineq', 'fun': flange_con},
        {'type': 'ineq', 'fun': flange_con_2},
        {'type': 'ineq', 'fun': width_con},
        {'type': 'ineq', 'fun': width_con_2},
        {'type': 'ineq', 'fun': width_con_3},
        {'type': 'ineq', 'fun': web_con},
        {'type': 'ineq', 'fun': web_con_2},
        {'type': 'ineq', 'fun': glue_con},
        {'type': 'ineq', 'fun': glue_con_2},
        {'type': 'ineq', 'fun': a_con},
        # {'type': 'eq', 'fun': packing_con}
    ]
    res = minimize(optimize, [width_guess, web_height_guess, flange_guess, glue_guess, a_guess], constraints=cons)
    solver = get_force_solver(*res.x)
    r = list(res.x[:])
    r.append(solver)
    return r

def search_pi(params, supported = False):
    width_guess, web_height_guess, flange_guess, glue_guess, a_guess = params
    *values, solver = optimize_pi(web_height_guess, flange_guess, glue_guess, a_guess, width_guess, 2, 2, supported = supported)
    failure_load, reason, section_num, section = solver.failure_load()
    values.insert(0, failure_load)
    values.append(solver)
    return values

def search_area_pi(width_mm, height_mm, flange_mm, glue_mm, a_mm, supported = False):
    max_area = 813 * 1016
    min_w, max_w, step_w = width_mm
    min_h, max_h, step_h = height_mm
    min_f, max_f, step_f = flange_mm
    min_g, max_g, step_g = glue_mm
    min_a, max_a, step_a = a_mm

    space_w = np.linspace(min_w, max_w, step_w)
    space_h = np.linspace(min_h, max_h, step_h)
    space_f = np.linspace(min_f, max_f, step_f)
    space_g = np.linspace(min_g, max_g, step_g)
    space_a = np.linspace(min_a, max_a, step_a)

    best = None
    count = 0
    total = len(space_w) * len(space_h) * len(space_f) * len(space_g) * len(space_a)
    for w in space_w:
        for h in space_h:
            for f in space_f:
                for g in space_g:
                    for a in space_a:
                        res = search_pi([w, h, f, g, a], supported = supported)
                        solver = res[-1]
                        packing_prop, diff = pack_rectangles(solver)
                        if packing_prop < 1:
                            print("Solution did not pack")
                        elif best is None:
                            best = res
                            print("Set Initial Best")
                            continue
                        elif res[0] > best[0]:
                            if packing_prop < 1:
                                print("New best did not pack")
                            elif solver.area() <= max_area + 0.001:
                                best = res
                                print("New Best:", round(max_area - solver.area()))
                            else:
                                print("New best was over area", round(max_area - solver.area()))
                        count += 1
                        print(f"{count}/{total} ", res[0], [w, h, f, g, a], res[1:-1])
    return best

if __name__ == "__main__":
    def summarize_pi(params, supported = False):
        width, web_height, flange_width, glue_width, a = params
        max_area = 813 * 1016
        section = PiSection(width, web_height, flange_width, glue_width, 2, 2, 1.27)
        # bridge = Bridge([(280 + 15, -0.5), (280 + 390 + 15, -0.5)], [15, 980-15], 980, 300)
        bridge = Bridge([(280 + 15, -0.5), (280 + 390 + 15, -0.5)], [15, 980-15], 980, a)
        extra_rects = []
        if supported:
            extra_rects = [(45, 600), (45, 600), (22.5, 600), (22.5, 600), (22.5, 600), (22.5, 600), (42.46, 600), (42.46, 600), (42.46, 600), (20.595, 600), (20.595, 600)]
            bridge.support_section()
        material = Material(30, 6, 4, 2, 4000, 0.2)
        solver = MaxForceSolver(section, material, bridge)
        max_force, reason, section_num, section= solver.failure_load()
        num_diaphragms = bridge.num_diaphragms()
        diaphragm_positions = [round(d, 2) for d in bridge.diaphragms]
        print(f"Best Width: {round(width, 2)},  Best Web Height: {round(web_height, 2)},  Best Flange Width: {round(flange_width, 2)},  Best Glue Width: {round(glue_width, 2)},  Best Number of Diaphragms: {num_diaphragms}")
        print(f"Diaphragm Positions: {diaphragm_positions}")
        print(f"Total Area: {round(solver.area(), 2)},  Area Left: {round(max_area - solver.area(), 2)}")
        failure_load, reason, section_num, section= solver.failure_load()
        print(f"Max Force: {round(failure_load, 2)}, Reason: {reason} on section {section_num}{section}\n")
        pack_rectangles(solver, extra_rectangles=extra_rects, outfile='out')

    def summarize_i(params, supported = True):
        top_width, top_glue_width, bottom_width, bottom_glue_width, web_width, web_height, a = params
        max_area = 813 * 1016
        section = ISection(top_width, 1, top_glue_width, bottom_width, 2, bottom_glue_width, web_width, web_height, 1.27)
        bridge = Bridge([(280 + 15, -0.5), (280 + 390 + 15, -0.5)], [15, 980-15], 980, a)
        extra_rects = []
        if supported:
            glued_width = 15
            extra_rects = [(45, 600), (45, 600), (22.5, 600), (22.5, 600), (22.5, 600), (22.5, 600), (42.46, 600), (42.46, 600), (42.46, 600), (20.595, 600), (20.595, 600)]
            # extra_rects = [(85, 600), (85, 600), (85 - 2*1.27, 600), (85 - 2*1.27, 600), (85 - 2*1.27, 600), ((85 - 3*1.27)/2, 600), ((85 - 3*1.27)/2, 600)]
            bridge.support_section()
        material = Material(30, 6, 4, 2, 4000, 0.2)
        solver = MaxForceSolver(section, material, bridge)
        max_force, reason, section_num, section= solver.failure_load()
        num_diaphragms = bridge.num_diaphragms()
        diaphragm_positions = [round(d, 2) for d in bridge.diaphragms]
        print(f"Top Width: {round(top_width, 2)},   Top Glue Width: {round(top_glue_width, 2)},   Bottom Width: {round(bottom_width, 2)},   Bottom Glue Width: {round(bottom_glue_width, 2)},   Web Width: {round(web_width, 2)},   Web Height: {round(web_height, 2)},   a: {round(a, 2)}")
        print(f"Diaphragm Positions: {diaphragm_positions}")
        print(f"Total Area: {round(solver.area(), 2)},  Area Left: {round(max_area - solver.area(), 2)}")
        failure_load, reason, section_num, section= solver.failure_load()
        print(f"Max Force: {round(failure_load, 2)}, Reason: {reason} on section {section_num}{section}\n")
        pack_rectangles(solver, extra_rectangles=extra_rects, outfile='out')

    # section = PiSection(120, 100, 20, 5, 2, 2, 1.27)
    # # bridge = Bridge([(280 + 15, -0.5), (280 + 390 + 15, -0.5)], [15, 980-15], 980, 300)
    # bridge = Bridge([(900, -1)], [0, 1200], 1200, 300)
    # material = Material(30, 6, 4, 2, 4000, 0.2)
    # solver = MaxForceSolver(section, material, bridge)
    # max_force, reason, section_num, section= solver.failure_load()
    # print(max_force, reason)

    # print(search_pi([100, 20, 5, 100, 120]))

    # max_force, *values, solver = search_area_pi([50, 150, 5], [130, 197, 2], [25, 40, 5], [10, 15, 5], [70, 120, 5], supported=True)
    # summarize_pi(values)

    # print("Optimizing I with both decks at 2 thick")
    # *values, solver = optimize_i(100, 6.35, 60.31, 6.35, 40.41, 108.66, 80)
    # summarize_i(values, supported=True)

    # top_width, top_glue_width, bottom_width, bottom_glue_width, web_width, web_height, a = params
    summarize_i([100, 6.35, 61.52, 6.35, 40.61, 119.56, 80], supported=True)

    # print("Searching I beam area")
    # failure_load, *params, solver = search_area_i((20, 50, 3), (40, 80, 4), (6.35, 6.35, 1), (20, 80, 2), (100, 190, 2), (80, 200, 2))
    # summarize_i(params, supported=True)

    # summarize_i([32.7, 6.35, 62.66, 6.35, 20, 181, 97.11], supported=True)

    # print("Optimizing With Deck and Flange at 2 thick")
    # *values, solver = optimize_pi(197.46, 39.57, 9.76, 95, 118.19, 2, 2, supported=True)
    # summarize_pi(values, supported=True)

    # print("Optimizing With Deck and Flange at 2 thick")
    # *values, solver = optimize_pi(197.46, 39.57, 7.12, 92.62, 127.63, 2, 2, supported=False)
    # summarize_pi(values, supported=False)

    # width, web_height, flange_width, glue_width, a = [150, 197, 25, 10.67, 10]
    # summarize_pi([width, web_height, flange_width, glue_width, a])

    # summarize_pi([133.25, 197.46, 39.57, 13.58, 95.92])
    # summarize_pi([150, 197, 25, 2.01, 10])
    # summarize_pi([118.19, 197.46, 39.57, 9.76, 93.25])
    # summarize_pi([83.09, 166.32, 34, 10, 90], supported=True)
    # summarize_pi([99.73, 164.63, 37.22, 6, 85], supported=True)

    # good_width = 118.19
    # good_web_height = 197.46
    # good_flange_width = 39.57
    # good_glue_width = 9.76
    # good_a = 93.25
    # deck_thickness = 2
    # *values, solver = optimize_pi(good_web_height, good_flange_width, good_glue_width, good_a, good_width, deck_thickness, deck_thickness)
    # summarize_pi(values)