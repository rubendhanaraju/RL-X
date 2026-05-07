from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as Et


class D3ILSceneBuilder:
    def __init__(self, assets_root):
        self.assets_root = Path(assets_root)
        self.root = Et.Element("mujoco")
        Et.SubElement(self.root, "worldbody")

    @property
    def worldbody(self):
        return self.root.find("worldbody")

    def body_elements(self):
        return list(self.worldbody)

    def add_body(self, body):
        self.worldbody.append(body)

    def add_primitive_body(self, name, geom_type, pos, quat, size, rgba, mass=0.1, static=False, visual_only=False):
        self.add_body(self.primitive_body(name, geom_type, pos, quat, size, rgba, mass, static, visual_only))

    def primitive_body(self, name, geom_type, pos, quat, size, rgba, mass=0.1, static=False, visual_only=False):
        body = Et.Element("body", {"name": name, "pos": self.vec_to_str(pos), "quat": self.vec_to_str(quat)})
        geom_attrs = {
            "type": geom_type,
            "name": f"{name}:geom",
            "size": self.vec_to_str(size),
            "rgba": self.vec_to_str(rgba),
        }
        if mass is not None:
            geom_attrs["mass"] = str(mass)
        if visual_only:
            geom_attrs["contype"] = "0"
            geom_attrs["conaffinity"] = "0"
        Et.SubElement(body, "geom", geom_attrs)
        if not static:
            Et.SubElement(body, "freejoint")
        return body

    def merge_object_xml(self, rel_path, pos=None, quat=None, freejoint=False, gravcomp=False):
        object_root = Et.parse(self.assets_root / rel_path).getroot()
        worldbody = object_root.find("worldbody")
        if worldbody is None:
            return
        body = worldbody.find("body")
        if body is not None:
            if pos is not None:
                body.set("pos", self.vec_to_str(pos))
            if quat is not None:
                body.set("quat", self.vec_to_str(quat))
            if gravcomp:
                body.set("gravcomp", "1")
            if freejoint and body.find("freejoint") is None and body.find("joint") is None:
                if body.find("inertial") is None:
                    body.insert(
                        0,
                        Et.Element(
                            "inertial",
                            {
                                "mass": "0.001",
                                "pos": "0 0 0",
                                "diaginertia": "1e-6 1e-6 1e-6",
                            },
                        ),
                    )
                body.insert(1, Et.Element("freejoint"))
        for child in list(worldbody):
            self.worldbody.append(deepcopy(child))

    def vec_to_str(self, values):
        return " ".join(str(v) for v in values)
