import taichi as ti
import sys
import colorama
import math
import numpy as np
import os as os
from pyevtk.hl import pointsToVTK

from config import NumericalSettings, PhysicalQuantities, GravityField
from fields import ParticleFields, GridFields, StabilizationFields, ProjectionFields, PenaltyMethodFields

physical = PhysicalQuantities()
numerical = NumericalSettings(physical)
gravitational = GravityField(numerical, physical)
particle = ParticleFields(numerical.numParticles, numerical.valueType)
grid = GridFields(numerical.numGrids, numerical.valueType)
stability = StabilizationFields(numerical.numGrids, numerical.valueType)
projection = ProjectionFields(numerical.numGrids, numerical.valueType)
penalty = PenaltyMethodFields(numerical.numCells, numerical.valueType)


@ti.func
def reproducingKernelFunction(position, base, kernelSupportSize, gridSpacing, maxNumof1DgridNodesWithinSupport, dimension, switch_kernelFunction):
    """
    The MATLAB version of Reproducing Kernel Function was written by Prof. Tsung-Hui Huang (contact: tsunghuihuang@ntu.edu.tw),
    please refer to this paper https://link.springer.com/article/10.1007/s40571-019-00272-x for additional details.
    The Taichi version of Reproducing Kernel Function was later written by Cameron Rodriguez (contact camjohnrod@gmail.com).
    
    Do not change the integers3 and 2 in the functions to variables, such as,
        maxNumof1DgridNodesWithinSupport_int = 3
        dimension_int = 2
    then calls
        kerneFunctionWeight1D = ti.Matrix.zero(numerical.valueType, maxNumof1DgridNodesWithinSupport_int, dimension_int)
    
    Taichi will not recognize them due to its compiler design pattern.
    """

    reproducingKernelFunction = ti.Matrix.zero(numerical.valueType, 3, 3)
    reproducingKernelFunction_xDerivative = ti.Matrix.zero(numerical.valueType, 3, 3)
    reproducingKernelFunction_yDerivative = ti.Matrix.zero(numerical.valueType, 3, 3)
    momentMatrix = ti.Matrix.zero(numerical.valueType, 3, 3)

    kerneFunctionWeight1D = ti.Matrix.zero(numerical.valueType, 3, 2)
    kerneFunctionWeight2D = ti.Matrix.zero(numerical.valueType, 3, 3)
    for i, d in ti.static(ti.ndrange(3, 2)):
        currentGridNodeLocation = float(i + base[d]) * gridSpacing
        if currentGridNodeLocation >= 0:
            normalizedParticle2GridNodeDistance = abs(position[d] - float(currentGridNodeLocation)) / kernelSupportSize

            if switch_kernelFunction:
                    if 0 <= normalizedParticle2GridNodeDistance and normalizedParticle2GridNodeDistance < 0.5:
                        kerneFunctionWeight1D[i, d] = 2/3 - 4*normalizedParticle2GridNodeDistance**2 + 4*normalizedParticle2GridNodeDistance**3
                    elif 0.5 <= normalizedParticle2GridNodeDistance and normalizedParticle2GridNodeDistance < 1:
                        kerneFunctionWeight1D[i, d] = 4/3 - 4*normalizedParticle2GridNodeDistance + 4*normalizedParticle2GridNodeDistance**2 - (4/3)*normalizedParticle2GridNodeDistance**3
            elif switch_kernelFunction:
                    if 0 <= normalizedParticle2GridNodeDistance and normalizedParticle2GridNodeDistance < 1:
                        kerneFunctionWeight1D[i, d] = normalizedParticle2GridNodeDistance - 1
                    elif 1 <= normalizedParticle2GridNodeDistance:
                        kerneFunctionWeight1D[i, d] = 0

    for i, j in ti.static(ti.ndrange(3, 3)):
        currentGridNodeLocation = [float(i + base[0]) * gridSpacing, float(j + base[1]) * gridSpacing]
        if currentGridNodeLocation[0] >= 0 and currentGridNodeLocation[1] >= 0:
            kerneFunctionWeight2D[i, j] = kerneFunctionWeight1D[i, 0] * kerneFunctionWeight1D[j, 1]
            # Define P(xi - xp)
            Pxi = ti.Vector([1.0, position[0] - currentGridNodeLocation[0], position[1] - currentGridNodeLocation[1]])
            if kerneFunctionWeight2D[i, j] != 0:
                # momentMatrix += weight[i, j] * Pxi @ Pxi.transpose()
                momentMatrix += kerneFunctionWeight2D[i, j] * Pxi.outer_product(Pxi)

    momentMatrix_inv = momentMatrix.inverse()

    for i, j in ti.static(ti.ndrange(3, 3)):
        currentGridNodeLocation = [float(i+base[0]) * gridSpacing, float(j+base[1]) * gridSpacing]
        if kerneFunctionWeight2D[i, j] != 0:
            if currentGridNodeLocation[0] >= 0 and currentGridNodeLocation[1] >= 0:
                # Define P(xi - xp)
                Pxi = ti.Vector([1.0, position[0] - currentGridNodeLocation[0], position[1] - currentGridNodeLocation[1]])
                Pxp = ti.Vector([1.0, position[0] - position[0], position[1] - position[1]])
                dPxpX = ti.Vector([0.0, -1.0, 0.0])
                dPxpY = ti.Vector([0.0, 0.0, -1.0])

                phi = kerneFunctionWeight2D[i, j] * (Pxp.dot(momentMatrix_inv @ Pxi))
                dphi_x1 = kerneFunctionWeight2D[i, j] * (dPxpX.dot(momentMatrix_inv @ Pxi))
                dphi_x2 = kerneFunctionWeight2D[i, j] * (dPxpY.dot(momentMatrix_inv @ Pxi))

                reproducingKernelFunction[i, j] = phi
                reproducingKernelFunction_xDerivative[i, j] = dphi_x1
                reproducingKernelFunction_yDerivative[i, j] = dphi_x2

    return reproducingKernelFunction, reproducingKernelFunction_xDerivative, reproducingKernelFunction_yDerivative

# penalty boundary that ensures the boundary conditions are embedded in the weak formulation
@ti.func
def penaltyBoundary(boundary, gridSpacing, gridNodeShift, kernelSupportSize, numCells, maxNumof1DgridNodesWithinSupport, dimension, switch_kernelFunction):
    for i in boundary:
        base = (boundary[i] / gridSpacing - gridNodeShift).cast(ti.i32)
        shapeFunction_grid, _, _ = reproducingKernelFunction(boundary[i], base, kernelSupportSize, gridSpacing, maxNumof1DgridNodesWithinSupport, dimension, switch_kernelFunction)
        
        s1, s2 = 0, 0
        if boundary == penalty.left_boundary or boundary == penalty.right_boundary:
            s1 = 1
            s2 = 0
        elif boundary == penalty.bottom_boundary or boundary == penalty.top_boundary:
            s1 = 0
            s2 = 1

        diagonalPenalty = ti.Matrix([[s1, 0], [0, s2]])

        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])

            if boundary == penalty.left_boundary:
                if float(base[0] + offset[0]) >= 2:
                    grid.mass_grid[base[0] + i, base[1] + j] += gridSpacing * numerical.penaltyParameter * shapeFunction_grid[i, j] * diagonalPenalty
                if float(base[0] + offset[0]) < 2:
                    grid.mass_grid[base[0] + i, base[1] + j] += gridSpacing * numerical.penaltyParameter * diagonalPenalty

            if boundary == penalty.right_boundary:
                if float(base[0] + offset[0]) <= numCells - 2:
                    grid.mass_grid[base[0] + i, base[1] + j] += gridSpacing * numerical.penaltyParameter * shapeFunction_grid[i, j] * diagonalPenalty
                if float(base[0] + offset[0]) > numCells - 2:
                    grid.mass_grid[base[0] + i, base[1] + j] += gridSpacing * numerical.penaltyParameter * diagonalPenalty

            if boundary == penalty.bottom_boundary:
                if float(base[1] + offset[1]) >= 2:
                    grid.mass_grid[base[0] + i, base[1] + j] += gridSpacing * numerical.penaltyParameter * shapeFunction_grid[i, j] * diagonalPenalty
                if float(base[1] + offset[1]) < 2:
                    grid.mass_grid[base[0] + i, base[1] + j] += gridSpacing * numerical.penaltyParameter * diagonalPenalty

            if boundary == penalty.top_boundary:
                if float(base[1] + offset[1]) <= numCells - 2:
                    grid.mass_grid[base[0] + i, base[1] + j] += gridSpacing * numerical.penaltyParameter * shapeFunction_grid[i, j] * diagonalPenalty
                if float(base[1] + offset[1]) > numCells - 2:
                    grid.mass_grid[base[0] + i, base[1] + j] += gridSpacing * numerical.penaltyParameter * diagonalPenalty

# a helper function to output vtk and png file names using scientific expression
def format_exp(x, n, d=6):
    significand = x / 10 ** n
    exp_sign = '+' if n >= 0 else ''
    return f'{significand:.{d}f}e{exp_sign}{n:02d}'

# a helper function to create vtk and png files. 
def createFilePaths(numerical: NumericalSettings):
    filepath = "mov"
    vtkpath = "vtk"

    filepath = filepath + "_dt" + format_exp(numerical.timeStep,math.floor(math.log10(numerical.timeStep)), 0)
    vtkpath = vtkpath + "_dt" + format_exp(numerical.timeStep, math.floor(math.log10(numerical.timeStep)), 0)

    if numerical.pressureMixingRatio == 1:
        filepath = filepath + "_mixed"
        vtkpath = vtkpath + "_mixed"
    elif numerical.pressureMixingRatio == 0:
        filepath = filepath + "_pointwise"
        vtkpath = vtkpath + "_pointwise"

    if numerical.switch_penaltyEBC:
        filepath = filepath + "_betaNor" + format_exp(numerical.penalty, math.floor(math.log10(numerical.penalty)), 0)
        vtkpath = vtkpath + "_betaNor" + format_exp(numerical.penalty, math.floor(math.log10(numerical.penalty)), 0)
    
    os.getcwd()
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    if not os.path.exists(vtkpath):
        os.makedirs(vtkpath)

    return filepath, vtkpath

# a progress bar for visualization on terminal when running this program.
def progressBar(progress, total, color=colorama.Fore.YELLOW):
    percentage = 100 * (progress/float(total))
    bar = '█' * int(percentage) + '-'*(100 - int(percentage))
    print(color + f"\r|{bar}| {percentage:.2f}%" +
          " | Current Time: " + str(progress))  # , end = "\r")
    sys.stdout.write("\033[F")  # back to previous line
    sys.stdout.write("\033[K")  # clear line
    if progress >= total:
        print(colorama.Fore.GREEN + f"\r|{bar}| {percentage:.2f}%" +
              " | It's done ! | Current time: " + str(progress), end="\r")
        print(colorama.Fore.RED)


positionX = ti.field(dtype=numerical.valueType, shape=(numerical.numParticlesX, numerical.numParticlesY))
positionY = ti.field(dtype=numerical.valueType, shape=(numerical.numParticlesX, numerical.numParticlesY))

# the initialization step that defines the initial particle physical properties.
@ti.kernel
def initialization():
    for i, j in positionX:
        positionX[i, j] = (i / (numerical.numParticlesX - 1)) * numerical.fluidWidth + 2 * numerical.gridSpacing
        positionY[i, j] = (j / (numerical.numParticlesY - 1)) * numerical.fluidHeight + 2 * numerical.gridSpacing

    for i in range(numerical.numParticles):
        col = i - numerical.numParticlesX * (i // numerical.numParticlesY)
        row = i // numerical.numParticlesX
        particle.position[i] = [positionX[col, row], positionY[col, row]]
        particle.position[i] = [ti.random() * numerical.fluidWidth + 2 * numerical.gridSpacing,
                                ti.random() * numerical.fluidHeight + 2 * numerical.gridSpacing]       # Random distribution
        particle.velocity[i] = [0, 0]
        particle.mass[i] = numerical.initialParticleVolume * physical.particleDensity
        particle.material_id[i] = 0
        particle.volume[i] = numerical.initialParticleVolume
        particle.deformation_gradient[i] = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])

    if numerical.switch_penaltyEBC:
        for i in penalty.left_boundary:
            penalty.left_boundary[i] = [2 * numerical.gridSpacing, (2.5 + i) * numerical.gridSpacing]
            penalty.right_boundary[i] = [numerical.domainLength + 2 * numerical.gridSpacing, (2.5 + i) * numerical.gridSpacing]
            penalty.bottom_boundary[i] = [(2.5 + i) * numerical.gridSpacing, 2 * numerical.gridSpacing]
            penalty.top_boundary[i] = [(2.5 + i) * numerical.gridSpacing, numerical.domainLength + 2 * numerical.gridSpacing]

# this is the main calculation function that runs the MPM P2G->G2P solver.
@ti.kernel
def calculation():
    # reset after every timestep
    for i, j in grid.mass_grid:
        grid.velocity_grid[i, j] = [0, 0]
        grid.velocity_grid_initial[i, j] = [0, 0]
        grid.mass_grid[i, j] = [[0, 0], [0, 0]]
        grid.volume_grid[i, j] = 0
        grid.pressure_grid[i, j] = 0

    if numerical.switch_penaltyEBC:
        penaltyBoundary(penalty.left_boundary, numerical.gridSpacing, numerical.gridNodeShift, numerical.kernelSupportSize,
                        numerical.numCells, numerical.maxNumof1DgridNodesWithinSupport, numerical.dimension, numerical.switch_kernelFunction)
        penaltyBoundary(penalty.right_boundary, numerical.gridSpacing, numerical.gridNodeShift, numerical.kernelSupportSize,
                        numerical.numCells, numerical.maxNumof1DgridNodesWithinSupport, numerical.dimension, numerical.switch_kernelFunction)
        penaltyBoundary(penalty.bottom_boundary, numerical.gridSpacing, numerical.gridNodeShift, numerical.kernelSupportSize,
                        numerical.numCells, numerical.maxNumof1DgridNodesWithinSupport, numerical.dimension, numerical.switch_kernelFunction)
        penaltyBoundary(penalty.top_boundary, numerical.gridSpacing, numerical.gridNodeShift, numerical.kernelSupportSize,
                        numerical.numCells, numerical.maxNumof1DgridNodesWithinSupport, numerical.dimension, numerical.switch_kernelFunction)

    # particle-to-grid projection
    for p in particle.position:
        # cellBase = (particle.position[p] * numerical.inverseGridSpacing).cast(ti.i32)
        # Define the bottom left corner of the surrounding 3x3 grid of neighboring nodes
        base = (particle.position[p] * numerical.inverseGridSpacing - numerical.gridNodeShift).cast(ti.i32)
        vector_base2CurrentParticle = (particle.position[p] / (numerical.gridSpacing +
                                       numerical.numericalTolerance) - base.cast(numerical.valueType)) * numerical.gridSpacing

        particle.partitionofUnity[p] = ti.cast(0, numerical.valueType)
        particle.consistency[p] = ti.cast(0, numerical.valueType)
        particle.consistency_dx[p] = ti.cast(0, numerical.valueType)
        particle.consistency_dy[p] = ti.cast(0, numerical.valueType)

        shapeFunction_grid, shapeFunction_grid_dx, shapeFunction_grid_dy = reproducingKernelFunction(
            particle.position[p], base, numerical.kernelSupportSize, numerical.gridSpacing, numerical.maxNumof1DgridNodesWithinSupport, numerical.dimension, numerical.switch_kernelFunction)

        for i, j in ti.static(ti.ndrange(3, 3)): # numerical.maxNumof1DgridNodesWithinSupport
            currentGridNodeLocation = [float(i + base[0]) * numerical.gridSpacing, float(j + base[1]) * numerical.gridSpacing]
            shapeFunction_grid_gradient = ti.Vector([shapeFunction_grid_dx[i, j], shapeFunction_grid_dy[i, j]])
            particle.partitionofUnity[p] += shapeFunction_grid[i, j]
            particle.consistency[p] += shapeFunction_grid[i, j] * currentGridNodeLocation[0] * currentGridNodeLocation[1]
            particle.consistency_dx[p] += shapeFunction_grid_gradient[0] * currentGridNodeLocation[0] * currentGridNodeLocation[1]
            particle.consistency_dy[p] += shapeFunction_grid_gradient[1] * currentGridNodeLocation[0] * currentGridNodeLocation[1]

            # Vector of grid node positions relative to "base"
            offset = ti.Vector([i, j])
            # A vector from the current grid node to the current particle
            dpos = offset.cast(numerical.valueType) * numerical.gridSpacing - vector_base2CurrentParticle

            # define the contribution of the velocity gradient to the particle momentum
            APIC_velocity_grid = particle.velocity_gradient[p] @ dpos
            grid.volume_grid[base[0] + i, base[1] + j] += shapeFunction_grid[i, j] * particle.volume[p]
            grid.mass_grid[base[0] + i, base[1] + j] += shapeFunction_grid[i, j] * particle.mass[p] * ti.Matrix.identity(numerical.valueType, 2)

            if numerical.switch_vt_I_APIC:
                grid.velocity_grid_initial[base[0] + i, base[1] + j] += shapeFunction_grid[i, j] * \
                    particle.mass[p] * (particle.velocity[p] + APIC_velocity_grid)
                grid.velocity_grid[base[0] + i, base[1] + j] += shapeFunction_grid[i, j] * \
                    particle.mass[p] * (particle.velocity[p] + APIC_velocity_grid)
            else:
                grid.velocity_grid_initial[base[0] + i, base[1] + j] += shapeFunction_grid[i, j] * particle.mass[p] * particle.velocity[p]
                grid.velocity_grid[base[0] + i, base[1] + j] += shapeFunction_grid[i, j] * particle.mass[p] * particle.velocity[p]

            grid.velocity_grid[base[0] + i, base[1] + j] += numerical.timeStep * (particle.volume[p] * shapeFunction_grid[i, j]
                                                                       * gravitational.gravityField[0] - particle.volume[p] * (particle.stress[p] @ shapeFunction_grid_gradient))
            grid.pressure_grid[base[0] + i, base[1] + j] += shapeFunction_grid[i, j] * particle.volume[p] * particle.pressure[p] - \
                numerical.timeStep * physical.bulkModulus * particle.volume[p] * shapeFunction_grid[i, j] * particle.divergenceofVelocity[p]

        particle.consistency[p] -= particle.position[p][0] * particle.position[p][1]
        particle.consistency_dx[p] -= float(1) * particle.position[p][1]
        particle.consistency_dy[p] -= float(1) * particle.position[p][0]

    # grid update
    for i, j in grid.mass_grid:
        if grid.volume_grid[i, j] != 0:
            grid.pressure_grid[i, j] /= grid.volume_grid[i, j]
        if grid.mass_grid[i, j][0, 0] != 0 and grid.mass_grid[i, j][1, 1] != 0:
            if grid.mass_grid[i, j].determinant() != 0:
                grid.velocity_grid[i, j] = grid.mass_grid[i, j].inverse() @ grid.velocity_grid[i, j]
            if numerical.switch_penaltyEBC:
                pass
            else:
                if i < numerical.maxNumof1DgridNodesWithinSupport and grid.velocity_grid[i, j][0] < 0:
                    grid.velocity_grid[i, j][0] = 0
                if i > numerical.numGrids - numerical.maxNumof1DgridNodesWithinSupport - 1 and grid.velocity_grid[i, j][0] > 0:
                    grid.velocity_grid[i, j][0] = 0
                if j < numerical.maxNumof1DgridNodesWithinSupport and grid.velocity_grid[i, j][1] < 0:
                    grid.velocity_grid[i, j][1] = 0
                if j > numerical.numGrids - numerical.maxNumof1DgridNodesWithinSupport - 1 and grid.velocity_grid[i, j][1] > 0:
                    grid.velocity_grid[i, j][1] = 0

    # grid-to-particle interpolation
    for p in particle.position:
        # the bottom left vertex of a 3x3 support for each particle
        base = (particle.position[p] * numerical.inverseGridSpacing - numerical.gridNodeShift).cast(ti.i32)
        vector_base2CurrentParticle = (particle.position[p] / (numerical.gridSpacing +
                                       numerical.numericalTolerance) - base.cast(numerical.valueType)) * numerical.gridSpacing

        velocity_APIC = ti.Vector.zero(numerical.valueType, 2)
        velocity_FLIP = ti.Vector.zero(numerical.valueType, 2)
        new_velocity_gradient = ti.Matrix.zero(numerical.valueType, 2, 2)
        new_divergenceofVelocity = ti.cast(0, numerical.valueType)
        new_pressure = ti.cast(0, numerical.valueType)

        shapeFunction_grid, shapeFunction_grid_dx, shapeFunction_grid_dy = reproducingKernelFunction(particle.position[p], base, numerical.kernelSupportSize, numerical.gridSpacing, numerical.maxNumof1DgridNodesWithinSupport, numerical.dimension, numerical.switch_kernelFunction)

        for i, j in ti.static(ti.ndrange(3, 3)): # numerical.maxNumof1DgridNodesWithinSupport
            shapeFunction_grid_gradient = ti.Vector([shapeFunction_grid_dx[i, j], shapeFunction_grid_dy[i, j]])
            offset = ti.Vector([i, j])
            new_velocity_gradient += grid.velocity_grid[base[0] + i, base[1] + j].outer_product(shapeFunction_grid_gradient)  # define the velocity gradient
            new_divergenceofVelocity += grid.velocity_grid[base[0] + i, base[1] + j].dot(shapeFunction_grid_gradient)

            velocity_APIC += shapeFunction_grid[i, j] * grid.velocity_grid[base[0] + i, base[1] + j]
            velocity_FLIP += shapeFunction_grid[i, j] * (grid.velocity_grid[base[0] + i, base[1] + j] - (grid.mass_grid[base[0] + i, base[1] + j].inverse() @ grid.velocity_grid_initial[base[0] + i, base[1] + j]))
            new_pressure += shapeFunction_grid[i, j] * grid.pressure_grid[base[0] + i, base[1] + j]

        velocity_FLIP += particle.velocity[p]
        particle.velocity[p] = numerical.flipBlendParameter * velocity_FLIP + (1 - numerical.flipBlendParameter) * velocity_APIC
        particle.position[p] += numerical.timeStep * particle.velocity[p]
        particle.velocity_gradient[p] = new_velocity_gradient
        particle.deformation_gradient[p] = (ti.Matrix.identity(numerical.valueType, 2) + numerical.timeStep * particle.velocity_gradient[p]) @ particle.deformation_gradient[p]

        # --- jacobian ---
        # method 1: Singular value decomposition
        # _, sig_F, _ = ti.svd(particle.deformation_gradient[p])
        # jacobian = ti.cast(1.0, numerical.valueType)
        # for d in ti.static(range(numerical.dimension)):
        #     jacobian *= sig_F[d, d]

        # method 2: direct calculation
        jacobian = particle.deformation_gradient[p][0, 0] * particle.deformation_gradient[p][1, 1] \
            - particle.deformation_gradient[p][1, 0] * particle.deformation_gradient[p][0, 1]
        # --- jacobian ---
        
        particle.determinant_of_deformation_gradient[p] = jacobian
        particle.volume[p] = numerical.initialParticleVolume * particle.determinant_of_deformation_gradient[p]
        if particle.determinant_of_deformation_gradient[p] != 0:
            particle.particleDensity[p] = physical.particleDensity / particle.determinant_of_deformation_gradient[p]
        particle.mass[p] = particle.volume[p] * particle.particleDensity[p]

        particle.divergenceofVelocity[p] = new_divergenceofVelocity
        particle.pressure[p] -= numerical.timeStep * physical.bulkModulus * particle.divergenceofVelocity[p]
        particle.pressure[p] = numerical.pressureMixingRatio * new_pressure + (1 - numerical.pressureMixingRatio) * particle.pressure[p]

        strainRate = 0.5 * (particle.velocity_gradient[p] + \
                            ti.Matrix([[particle.velocity_gradient[p][0, 0], particle.velocity_gradient[p][1, 0]], \
                                       [particle.velocity_gradient[p][0, 1], particle.velocity_gradient[p][1, 1]]]))

        particle.stress[p] = - particle.pressure[p] * ti.Matrix.identity(numerical.valueType, 2) + 2 * physical.dynamicViscosity * strainRate

# post processing, to render the calculation results into vtk and png files.
# visualize vtk files in Paraview. 
def post_process(num_p, gui, vtkpath, filepath, num_calculation, count):
    xCoordinate = np.zeros(num_p)
    yCoordinate = np.zeros(num_p)
    zCoordinate = np.zeros(num_p)
    v_x = np.zeros(num_p)
    v_y = np.zeros(num_p)
    v_z = np.zeros(num_p)
    materialID = np.zeros(num_p)
    deformation_step = np.zeros(num_p)
    PoU = np.zeros(num_p)
    Consistency = np.zeros(num_p)
    Consistency_Gradx = np.zeros(num_p)
    Consistency_Grady = np.zeros(num_p)
    vp_mag_step = np.zeros(num_p)
    pressure = np.zeros(num_p)
    sigma_11 = np.zeros(num_p)
    sigma_22 = np.zeros(num_p)
    sigma_12 = np.zeros(num_p)
    det_deformation_gradient = np.zeros(num_p)

    xCoordinate[:] = particle.position.to_numpy()[:, 0]
    yCoordinate[:] = particle.position.to_numpy()[:, 1]
    zCoordinate[:] = 0 * particle.position.to_numpy()[:, 1]
    materialID[:] = particle.material_id.to_numpy()[:]
    deformation_step[:] = particle.determinant_of_deformation_gradient.to_numpy()[:]
    v_x[:] = particle.velocity.to_numpy()[:, 0]
    v_y[:] = particle.velocity.to_numpy()[:, 1]
    v_z[:] = 0 * particle.velocity.to_numpy()[:, 1]
    pressure[:] = - (particle.stress.to_numpy()[:, 0, 0] +
                     particle.stress.to_numpy()[:, 1, 1]) / 3
    sigma_11[:] = particle.stress.to_numpy()[:, 0, 0]
    sigma_22[:] = particle.stress.to_numpy()[:, 1, 1]
    sigma_12[:] = particle.stress.to_numpy()[:, 0, 1]
    PoU[:] = particle.partitionofUnity.to_numpy()[:] - float(1.0)
    Consistency[:] = particle.consistency.to_numpy()[:]
    Consistency_Gradx[:] = particle.consistency_dx.to_numpy()[:]
    Consistency_Grady[:] = particle.consistency_dy.to_numpy()[:]
    det_deformation_gradient[:] = particle.determinant_of_deformation_gradient.to_numpy()[:]

    # Save data to VTK less frequently
    if count % (num_calculation) == 0:  # Save every 100th frame (adjust as needed)
        pointsToVTK(
            f'./{vtkpath}/points{gui.frame:06d}',
            xCoordinate, yCoordinate, zCoordinate,
            data={
                "ID": np.ascontiguousarray(materialID),
                "simgaxx": sigma_11,
                "simgayy": sigma_22,
                "simgaxy": sigma_12,
                "v_x": v_x,
                "v_y": v_y,
                "v_z": v_z,
                "Pressure": pressure,
                "Partition of Unity": PoU,
                "Consistency": Consistency,
                "Gradx Consistency": Consistency_Gradx,
                "Grady Consistency": Consistency_Grady,
                "Deformation": np.ascontiguousarray(det_deformation_gradient),
                "Velocity Mag": vp_mag_step
            }
        )

    # pointsToVTK(f'./{vtkpath}/points{gui.frame:06d}', xCoordinate, yCoordinate, zCoordinate,
    #             data={"ID": materialID, "simgaxx": sigma_11, "simgayy": sigma_22, "simgaxy": sigma_12,
    #                   "v_x": v_x, "v_y": v_y, "v_z": v_z, "Pressure": pressure, "Partition of Unity": PoU, "Consistency": Consistency,
    #                   "Gradx Consistency": Consistency_Gradx, "Grady Consistency": Consistency_Grady, "Deformation": deformation_step,
    #                   "Velocity Mag": vp_mag_step},
    #             fieldData={"Velocity": np.concatenate((v_x, v_y, v_z), axis=0)})
    if count % (num_calculation) == 0:
        colors = np.array([0x000000] * len(materialID), dtype=np.uint32)
        scaled_position = particle.position.to_numpy() / 0.5  # Scaling the coordinates to fit the 1 by 1 window
        gui.circles(scaled_position, radius=0.8, color=colors)
        gui.show(filepath + f'/{gui.frame:06d}.png')

    # colors = np.array([0x068587, 0xED553B, 0xEEEEF0], dtype=np.uint32)
    # gui.circles(particle.position.to_numpy(), radius=0.8, color=colors[particle.material_id.to_numpy()])
    # gui.show(f'{filepath}/{gui.frame:06d}.png')