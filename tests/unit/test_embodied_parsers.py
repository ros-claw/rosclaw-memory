"""
Unit tests for multi-format robot model parsers.

No external files or ROS toolchain required — all tests use inline XML.
"""

import pytest

from powermem.embodied.parsers import parse_model, get_parser
from powermem.embodied.parsers.base import ParseResult
from powermem.embodied.parsers.urdf_parser import URDFParser
from powermem.embodied.parsers.mjcf_parser import MJCFParser
from powermem.embodied.parsers.sdf_parser import SDFParser
from powermem.embodied.parsers.xacro_parser import XacroParser
from powermem.embodied.parsers.usd_parser import USDParser


# ---------------------------------------------------------------------------
# URDF
# ---------------------------------------------------------------------------

URDF_SIMPLE = """\
<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <geometry>
        <mesh filename="package://test/mesh.dae"/>
      </geometry>
    </visual>
  </link>
  <link name="arm_link">
    <inertial>
      <mass value="1.5"/>
      <origin xyz="0.5 0 0" rpy="0 0 0"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/>
    </inertial>
  </link>
  <joint name="shoulder" type="revolute">
    <parent link="base_link"/>
    <child link="arm_link"/>
    <origin xyz="0 0 1.0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" velocity="2.0" effort="50"/>
  </joint>
  <material name="red">
    <color rgba="1 0 0 1"/>
  </material>
</robot>
"""


class TestURDFParser:
    def test_parse_simple(self):
        parser = URDFParser()
        result = parser.parse(URDF_SIMPLE)
        assert result.format == "urdf"
        assert len(result.links) == 2
        assert len(result.joints) == 1
        assert result.dynamics.joint_names == ["shoulder"]
        assert result.dynamics.link_masses == [2.0, 1.5]
        assert len(result.mesh_paths) == 1
        assert "package://test/mesh.dae" in result.mesh_paths
        assert len(result.materials) == 1
        assert result.materials[0]["name"] == "red"

    def test_joint_limits(self):
        parser = URDFParser()
        result = parser.parse(URDF_SIMPLE)
        jl = result.dynamics.joint_limits[0]
        assert jl.min_rad == pytest.approx(-1.57)
        assert jl.max_rad == pytest.approx(1.57)
        assert jl.max_vel == 2.0
        assert jl.max_torque == 50.0

    def test_dh_params(self):
        parser = URDFParser()
        result = parser.parse(URDF_SIMPLE)
        dh = result.dynamics.dh_params[0]
        # origin xyz="0 0 1.0" => d=1.0, a=0
        assert dh.d == pytest.approx(1.0)
        assert dh.a == pytest.approx(0.0)

    def test_parse_invalid_xml(self):
        parser = URDFParser()
        result = parser.parse("not xml")
        assert len(result.warnings) > 0

    def test_parse_file(self, tmp_path):
        fpath = tmp_path / "robot.urdf"
        fpath.write_text(URDF_SIMPLE)
        result = parse_model(str(fpath))
        assert result.format == "urdf"
        assert result.source_path == str(fpath)


# ---------------------------------------------------------------------------
# MJCF
# ---------------------------------------------------------------------------

MJCF_SIMPLE = """\
<mujoco model="test">
  <compiler angle="degree"/>
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <inertial mass="2.0" pos="0 0 0" diaginertia="0.01 0.01 0.01"/>
      <geom type="box" size="0.1 0.1 0.1"/>
      <body name="arm" pos="0.5 0 0">
        <inertial mass="1.5" pos="0 0 0" diaginertia="0.005 0.005 0.005"/>
        <joint name="shoulder" type="hinge" axis="0 0 1" range="-90 90"/>
      </body>
    </body>
  </worldbody>
  <asset>
    <mesh name="mesh1" file="mesh.stl"/>
    <material name="mat1" rgba="1 0 0 1"/>
  </asset>
</mujoco>
"""


class TestMJCFParser:
    def test_parse_simple(self):
        parser = MJCFParser()
        result = parser.parse(MJCF_SIMPLE)
        assert result.format == "mjcf"
        # base + arm = 2 links
        assert len(result.links) >= 2
        assert len(result.dynamics.joint_names) >= 1
        assert "shoulder" in result.dynamics.joint_names

    def test_angle_conversion(self):
        parser = MJCFParser()
        result = parser.parse(MJCF_SIMPLE)
        # range="-90 90" with degree compiler => should convert to radians
        jl = result.dynamics.joint_limits[0]
        assert jl.min_rad < 0
        assert jl.max_rad > 0

    def test_mesh_from_asset(self):
        parser = MJCFParser()
        result = parser.parse(MJCF_SIMPLE)
        assert "mesh.stl" in result.mesh_paths

    def test_material_from_asset(self):
        parser = MJCFParser()
        result = parser.parse(MJCF_SIMPLE)
        mat_names = [m["name"] for m in result.materials]
        assert "mat1" in mat_names

    def test_collision_geoms_from_geom(self):
        mjcf_with_geom = """\
<mujoco model="test">
  <worldbody>
    <body name="base" pos="0 0 0">
      <geom type="sphere" size="0.1" pos="0 0 0.5"/>
      <geom type="capsule" size="0.05" fromto="0 0 0 0 0 1"/>
      <geom type="box" size="0.1 0.2 0.3" pos="1 0 0"/>
    </body>
  </worldbody>
</mujoco>
"""
        parser = MJCFParser()
        result = parser.parse(mjcf_with_geom)
        assert len(result.links) == 1
        geoms = result.links[0]["collision_geoms"]
        assert len(geoms) == 3
        types = {g["type"] for g in geoms}
        assert types == {"sphere", "capsule", "aabb"}


# ---------------------------------------------------------------------------
# SDF
# ---------------------------------------------------------------------------

SDF_SIMPLE = """\
<?xml version="1.0"?>
<sdf version="1.6">
  <model name="test_model">
    <link name="base">
      <inertial>
        <mass>2.0</mass>
        <pose>0 0 0.5 0 0 0</pose>
        <inertia>
          <ixx>0.01</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>0.01</iyy><iyz>0</iyz><izz>0.01</izz>
        </inertia>
      </inertial>
      <visual>
        <geometry>
          <mesh><uri>model://test/mesh.dae</uri></mesh>
        </geometry>
      </visual>
    </link>
    <link name="arm">
      <inertial>
        <mass>1.5</mass>
        <pose>0.5 0 0 0 0 0</pose>
      </inertial>
    </link>
    <joint name="shoulder" type="revolute">
      <parent>base</parent>
      <child>arm</child>
      <pose>0 0 1.0 0 0 0</pose>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>1.57</upper>
          <velocity>2.0</velocity>
          <effort>50</effort>
        </limit>
      </axis>
    </joint>
  </model>
</sdf>
"""


class TestSDFParser:
    def test_parse_simple(self):
        parser = SDFParser()
        result = parser.parse(SDF_SIMPLE)
        assert result.format == "sdf"
        assert len(result.links) == 2
        assert len(result.joints) == 1
        assert result.dynamics.joint_names == ["shoulder"]

    def test_mesh_uri(self):
        parser = SDFParser()
        result = parser.parse(SDF_SIMPLE)
        assert "model://test/mesh.dae" in result.mesh_paths


# ---------------------------------------------------------------------------
# Xacro
# ---------------------------------------------------------------------------

XACRO_SIMPLE = """\
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="xacro_bot">
  <xacro:property name="mass" value="3.0"/>
  <xacro:property name="length" value="0.5"/>

  <xacro:macro name="link_macro" params="name mass_val">
    <link name="${name}">
      <inertial>
        <mass value="${mass_val}"/>
        <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
      </inertial>
    </link>
  </xacro:macro>

  <xacro:link_macro name="base" mass_val="${mass}"/>
  <xacro:link_macro name="arm" mass_val="1.5"/>

  <joint name="shoulder" type="revolute">
    <parent link="base"/>
    <child link="arm"/>
    <limit lower="-1.57" upper="1.57" velocity="2.0" effort="50"/>
  </joint>
</robot>
"""


class TestXacroParser:
    def test_expand_properties(self):
        parser = XacroParser()
        expanded = parser.expand(XACRO_SIMPLE)
        assert "<mass value=\"3.0\"/>" in expanded
        assert "xacro:property" not in expanded

    def test_expand_macros(self):
        parser = XacroParser()
        expanded = parser.expand(XACRO_SIMPLE)
        assert '<link name="base"' in expanded
        assert '<link name="arm"' in expanded
        assert "xacro:macro" not in expanded
        assert "xacro:link_macro" not in expanded

    def test_parse_result(self):
        result = parse_model(XACRO_SIMPLE, fmt="xacro")
        assert result.format == "xacro"
        assert len(result.links) == 2
        assert len(result.joints) == 1
        assert result.dynamics.link_masses == [3.0, 1.5]

    def test_math_expression(self):
        text = """\
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:property name="a" value="2"/>
  <xacro:property name="b" value="3"/>
  <link name="test">
    <mass value="${a + b}"/></link>
</robot>
"""
        parser = XacroParser()
        expanded = parser.expand(text)
        assert '<mass value="5.0"/>' in expanded or '<mass value="5"/>' in expanded

    def test_condition_if(self):
        text = """\
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:property name="flag" value="1"/>
  <xacro:if value="${flag}">
    <link name="included"/>
  </xacro:if>
  <xacro:unless value="${flag}">
    <link name="excluded"/>
  </xacro:unless>
</robot>
"""
        parser = XacroParser()
        expanded = parser.expand(text)
        assert '<link name="included"/>' in expanded
        assert '<link name="excluded"/>' not in expanded


# ---------------------------------------------------------------------------
# USD
# ---------------------------------------------------------------------------

USD_ASCII = """\
#usda 1.0
(
    defaultPrim = "robot"
)

def Xform "robot"
{
    def PhysicsRevoluteJoint "shoulder"
    {
        uniform token axis = "Z"
        float2 range = (-1.57, 1.57)
        rel physics:body0 = </body0>
        rel physics:body1 = </body1>
    }
    def PhysicsRigidBody "base"
    {
        float mass = 2.0
    }
    def PhysicsRigidBody "arm"
    {
        float mass = 1.5
    }
}
"""


class TestUSDParser:
    def test_fallback_parse(self):
        parser = USDParser()
        result = parser.parse(USD_ASCII)
        assert result.format == "usd"
        assert len(result.warnings) >= 1  # usd-core not installed warning
        assert "shoulder" in result.dynamics.joint_names
        assert len(result.dynamics.link_masses) >= 2

    def test_mesh_refs(self):
        usd_with_mesh = """\
#usda 1.0
prepend references = @model://test/mesh.usd@
"""
        parser = USDParser()
        result = parser.parse(usd_with_mesh)
        assert "model://test/mesh.usd" in result.mesh_paths


# ---------------------------------------------------------------------------
# Registry / Auto-detect
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_get_parser_urdf(self):
        p = get_parser("urdf")
        assert p is not None
        assert isinstance(p, URDFParser)

    def test_get_parser_mjcf(self):
        p = get_parser("mjcf")
        assert p is not None
        assert isinstance(p, MJCFParser)

    def test_get_parser_sdf(self):
        p = get_parser("sdf")
        assert p is not None
        assert isinstance(p, SDFParser)

    def test_get_parser_xacro(self):
        p = get_parser("xacro")
        assert p is not None
        assert isinstance(p, XacroParser)

    def test_get_parser_usd(self):
        p = get_parser("usd")
        assert p is not None
        assert isinstance(p, USDParser)

    def test_auto_detect_urdf(self):
        result = parse_model(URDF_SIMPLE)
        assert result.format == "urdf"

    def test_auto_detect_mjcf(self):
        result = parse_model(MJCF_SIMPLE)
        assert result.format == "mjcf"

    def test_auto_detect_sdf(self):
        result = parse_model(SDF_SIMPLE)
        assert result.format == "sdf"

    def test_auto_detect_usd(self):
        result = parse_model(USD_ASCII)
        assert result.format == "usd"

    def test_unsupported_format(self):
        with pytest.raises(ValueError):
            parse_model("hello", fmt="unknown")

    def test_parse_result_to_dict(self):
        parser = URDFParser()
        result = parser.parse(URDF_SIMPLE)
        d = result.to_dict()
        assert d["format"] == "urdf"
        assert "dynamics" in d
        assert "links" in d
        assert "joints" in d

    def test_collision_geoms_extracted(self):
        urdf_with_collision = """\
<?xml version="1.0"?>
<robot name="test">
  <link name="base">
    <collision>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.1" length="0.4"/>
      </geometry>
    </collision>
    <collision>
      <origin xyz="0.2 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.05"/>
      </geometry>
    </collision>
  </link>
</robot>
"""
        parser = URDFParser()
        result = parser.parse(urdf_with_collision)
        assert len(result.links) == 1
        geoms = result.links[0]["collision_geoms"]
        assert len(geoms) == 2
        assert geoms[0]["type"] == "capsule"
        assert geoms[0]["radius"] == pytest.approx(0.1)
        assert geoms[1]["type"] == "sphere"
        assert geoms[1]["radius"] == pytest.approx(0.05)
