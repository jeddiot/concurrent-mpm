import taichi as ti


class PhysicalQuantities:
    def __init__(self):
        self.dynamicViscosity = 1e-3  # [Pa·s]
        self.poissonsRatio = 4.999e-1  # unitless
        self.bulkModulus = 2e6  # [Pa]
        self.youngsModulus = self.bulkModulus * 2 * (1 - self.poissonsRatio)  # [Pa]
        self.shearModulus = self.bulkModulus * (1 - self.poissonsRatio) / (1 + self.poissonsRatio)  # [Pa]
        self.particleDensity = 1000  # [kg/m³]
        self.gravity = -9.81  # [m/s²]


class NumericalSettings:
    def __init__(self, physical: PhysicalQuantities):
        self.valueType = ti.f64
        self.switch_vt_I_APIC = True  # True: velocity APIC, False: velocity PIC
        self.switch_overlineF = False  # F-Bar pressure stabilization
        self.switch_penaltyEBC = False
        self.switch_kernelFunction = True  # True bspline, else tent
        self.dimension = 2
        self.numericalTolerance = 1e-15
        self.simulationTime = 3
        self.totalTime = 0e-15
        self.timeStep = 2e-6
        self.penalty = 1e6
        self.pressureMixingRatio = 0  # 1mixed, 0pt
        self.flipBlendParameter = 0 # 1flp, 0apic
        self.numParticlesX = 149*2
        self.numParticlesY = 149
        self.numParticles = self.numParticlesX * self.numParticlesY
        self.domainLength = 3.22  # [m]
        self.fluidWidth = 1.2  # [m]
        self.fluidHeight = 0.6  # [m]
        self.initialParticleVolume = (self.fluidWidth * self.fluidHeight) / self.numParticles
        self.numGrids = 165
        self.numCells = self.numGrids - 1
        self.gridSpacing = self.domainLength / float(self.numCells - 4)
        self.inverseGridSpacing = 1 / self.gridSpacing
        self.kernelSupportSizeNormalized = 1.5
        self.kernelSupportSize = self.kernelSupportSizeNormalized * self.gridSpacing
        self.maxNumof1DgridNodesWithinSupport = int(self.kernelSupportSize * self.inverseGridSpacing * 2 + self.numericalTolerance)
        self.gridNodeShift = float(self.kernelSupportSizeNormalized - 1.0)
        self.penaltyParameter = self.penalty * physical.particleDensity * self.gridSpacing**2
        self.frameRate = 1e-2  # save every 100th frame (adjust as needed)


class GravityField:
    def __init__(self, numerical: NumericalSettings, physical: PhysicalQuantities):
        self.gravityField = ti.Vector.field(2, dtype=numerical.valueType, shape=(1, ))
        self.gravityField[0] = [0, numerical.fluidWidth * numerical.fluidHeight * physical.particleDensity * physical.gravity]


class SimulationConfig:
    def __init__(self):
        # Initialize all classes
        self.numericalSettings = NumericalSettings()
        self.physicalQuantities = PhysicalQuantities()
        self.gravityField = GravityField(self.numericalSettings.fluidWidth,
                                         self.numericalSettings.fluidHeight,
                                         self.physicalQuantities.particleDensity,
                                         self.physicalQuantities.gravity)

    def __repr__(self):
        return f"SimulationConfig({vars(self.numericalSettings)}, {vars(self.physicalQuantities)}, {vars(self.gravityField)})"


physical = PhysicalQuantities()
numerical = NumericalSettings(physical)
gravatational = GravityField(numerical, physical)
