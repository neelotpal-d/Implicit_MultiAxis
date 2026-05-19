import numpy as np
import pyvista as pv
import igl
import torch
import torch.nn as nn
import models_3DP as models
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from itertools import cycle
from siren_pytorch import SirenNet, Sine
from coupledSiren import scalarNet, DynamicLinearNet, scalarNet2
from collisionLoss import collison_loss


np.bool = np.bool_
device = 'cuda'



'''
DATA BLOCK

'''

########## DATA MANAGEMENT BLOCK##################


model = 'TshapeBracketNew'

mesh = pv.read("./inputs/TshapeBracketNew.obj")
meshVol = pv.read("./inputs/TshapeBracketNew.ele")

if model == 'topopt':
    mesh = pv.read("./inputs/topopt.obj")
    meshVol = pv.read("./inputs/topopt.ele")



meshVertices = np.array(mesh.points)
meshFaces = np.array(mesh.faces).reshape((-1,4))
meshFaces = meshFaces[:,1:4]


x_min,y_min,z_min = np.min(meshVertices, axis=0)
x_max,y_max,z_max = np.max(meshVertices, axis=0)



'''
x_vals = np.arange(x_min,x_max,0.03)
y_vals = np.arange(y_min,y_max,0.03)
z_vals = np.arange(z_min,z_max,0.03)

'''
x_vals = np.arange(x_min-1.0,x_max+1.0,2.0)
y_vals = np.arange(y_min-1.0,y_max+1.0,2.0)
z_vals = np.arange(z_min-1.0,z_max+1.0,2.0)



X,Y,Z = np.meshgrid(x_vals, y_vals, z_vals)

X = X.flatten()
Y = Y.flatten()
Z = Z.flatten()


meshGrid = pv.StructuredGrid(X,Y,Z)

pl1 = pv.Plotter()
#pl1.add_mesh(meshGrid)
#pl1.add_mesh(mesh)



sdf = igl.signed_distance(meshGrid.points, meshVertices, meshFaces)[0]
mask = sdf<0.5
maskedPoints = meshGrid.points[mask]
maskedGrid = pv.PolyData(maskedPoints)
#pl1.add_mesh(maskedGrid)
#pl1.show()
mask_out = sdf>0.5


rand_indices = torch.randperm(meshGrid.points[mask_out].shape[0])
domainPoints = torch.tensor(meshGrid.points[mask_out][rand_indices][0:12000], dtype=torch.float32, requires_grad=True, device = device)


###########################################################################






################# READ STRESS INPUT ####################################


file_path = './stress_TshapeBracketNew.txt'
if model == 'topopt':
    file_path = './stress_topopt.txt'

# Initialize an empty list to store the vectors
StressConsts = []

# Open the file for reading
with open(file_path, 'r') as file:
    for line in file:
        # Split the line into components based on spaces
        components = line.split()
        
        # Convert the components from strings to floats
        vector = [float(component) for component in components]
        
        # Ensure there are exactly 9 components
        if len(vector) != 9:
            raise ValueError("Each row must contain exactly 9 values.")
        
        # Append the vector to the list
        StressConsts.append(vector)

# Optionally, convert the list of vectors to a NumPy array for easier manipulation
StressConsts_np = np.array(StressConsts)

stressPoints = StressConsts_np[:,0:3]
maxStressDirns = StressConsts_np[:,3:6]
minStressDirns = StressConsts_np[:,6:9]


sPoints = torch.tensor(stressPoints, dtype=torch.float32, requires_grad=True, device=device)
minStressNormals = torch.tensor(minStressDirns, dtype=torch.float32, device= device)
maxStressNormals = torch.tensor(maxStressDirns, dtype=torch.float32, device= device)


mesh.compute_normals(cell_normals=False, inplace=True)
# ##########override stress#################
# sPoints = mesh.points
# sPointsX = sPoints[:,0]
# sPointsY = sPoints[:,1]
# sPointsZ = sPoints[:,2]
# X_mask = (sPointsX>x_min+0.1)&(sPointsX<x_max-0.1)
# Z_mask = (sPointsZ>z_min+0.1)&(sPointsZ<z_max-0.1)
# sPoints = sPoints[X_mask&Z_mask]
# s_normals = mesh.point_data['Normals']
# s_normals = torch.tensor(s_normals, dtype= torch.float32, device=device)
# s_xDir = torch.tensor([1,0,0], dtype= torch.float32, device=device)
# s_xDir = s_xDir.repeat(s_normals.shape[0],1)
# maxStressNormals = torch.cross(s_xDir, s_normals).detach()
# maxStressNormals = s_xDir.repeat(sPoints.shape[0],1).detach()
# maxStressNormals = maxStressNormals[X_mask&Z_mask]
# ########################################################

boundaryPoints_np = mesh.points
boundaryNormals_np = mesh.point_data['Normals']
rand_indices = torch.randperm(boundaryPoints_np.shape[0]).detach()
boundaryPoints_np = boundaryPoints_np[rand_indices]
boundaryNormals_np = boundaryNormals_np[rand_indices]

meshZpoints = boundaryPoints_np[:,1]
modelBaseMask = meshZpoints < (y_min+2.5)
modelNonBaseMask = meshZpoints >= (y_min-2.5)


basePoints = torch.tensor(boundaryPoints_np[modelBaseMask], dtype= torch.float32, device= device, requires_grad=True)


boundaryPoints = torch.tensor(boundaryPoints_np[modelNonBaseMask], dtype= torch.float32, device= device, requires_grad=True)
boundaryNormals = torch.tensor(boundaryNormals_np[modelNonBaseMask], dtype=torch.float32, device=device)


'''


boundaryPoints = torch.tensor(boundaryPoints_np, dtype= torch.float32, device= device, requires_grad=True)
boundaryNormals = torch.tensor(boundaryNormals_np, dtype=torch.float32, device=device)
'''



##########################################################################






###################### Connected Boundary Points #########################

# Ensure the mesh is clean and triangulated to handle any issues
mesh.clean(inplace=True)
mesh = mesh.triangulate()

# Initialize a dictionary to store connected points
connected_points = {i: set() for i in range(mesh.n_points)}

# Iterate over each cell to build the connectivity
for i in range(mesh.n_cells):
    cell = mesh.get_cell(i)  # Get the i-th cell
    points_in_cell = cell.point_ids  # Indices of points in the cell
    for point in points_in_cell:
        connected_points[point].update(points_in_cell)
        connected_points[point].discard(point)  # Remove itself

# Create a list of arrays for neighboring points
neighboring_points = []
for i in range(mesh.n_points):
    neighbors = list(connected_points[i])  # Get the connected point indices
    neighboring_points.append(mesh.points[neighbors])  # Append their coordinates

# Find the maximum number of neighbors
max_neighbors = max(len(neighbors) for neighbors in neighboring_points)

# Create an n x m x 3 array, filling missing neighbors with any existing neighbors
n = mesh.n_points
m = max_neighbors
neighboring_points_array = np.zeros((n, m, 3))

for i in range(n):
    neighbors = neighboring_points[i]
    num_neighbors = len(neighbors)
    neighboring_points_array[i, :num_neighbors, :] = neighbors  # Fill with neighbors
    if num_neighbors < m:
        # Fill remaining slots with any existing neighbors
        fill_value = neighbors[0] if num_neighbors > 0 else mesh.points[i]  # Use first neighbor or itself
        neighboring_points_array[i, num_neighbors:, :] = fill_value


meshZpoints = mesh.points[:,1]
modelBaseMask = meshZpoints < (y_min+2.5)
modelNonBaseMask = meshZpoints >= (y_min+2.5)

connect_boundaryPoints = torch.tensor(mesh.points[modelNonBaseMask], dtype= torch.float32, device= device, requires_grad=True)
connect_connectPoints = torch.tensor(neighboring_points_array[modelNonBaseMask], dtype= torch.float32, device= device, requires_grad=True)
connect_boundaryNormals = torch.tensor(mesh.point_data['Normals'][modelNonBaseMask], dtype= torch.float32, device= device, requires_grad=True)
#####################################################################################################



















#############################Monotonocity Loss###########################
class obstacleFunction(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,points): #(x,y,z)
        sdf = 1.02 -points[:,0]
        
        return sdf




def getMaxLaplacaian(points, model):
    model.eval()
    out = model(points)
    out2x = out['HX2'][:,0]
    out2y = out['HY2'][:,1]
    out2z = out['HZ2'][:,2]
    
    lap = out2x*out2x + out2y*out2y + out2z*out2z
    lap = torch.abs(lap)
    maxLap = torch.max(lap)
    
    model.train()
    
    return maxLap


def getMaxNormal(points, model):
    model.eval()
    out = model(points)
    grads = out['grads']
    gradsNorm = torch.norm(grads, dim=1)
        
    maxNorm = torch.max(gradsNorm)
    
    model.train()
    
    return maxNorm


def getMinNormal(points, model):
    model.eval()
    out = model(points)
    grads = out['grads']
    gradsNorm = torch.norm(grads, dim=1)
        
    minNorm = torch.min(gradsNorm)
    
    model.train()
    
    return minNorm

def detectSingularity():
    pass


def envIntersectionLoss(points,grads,model):
    model.eval()
    gradNorms = (torch.norm(grads, dim=1)).unsqueeze(1)
    gradDirs = grads/(gradNorms+1e-8)
    gradDirs = gradDirs*0.1
    
    
    p0 = points.detach()

    p1 = p0+0.5*gradDirs
    p2 = p0+1*gradDirs    
    p3 = p0+3*gradDirs
    p4 = p0+10*gradDirs
    
    new_points = torch.hstack((p1,p3,p4))
    outScalars = model(new_points)
    
    scalarDiff = 0.2 - outScalars
    #print(scalarDiff.shape)
    scalarDiff = torch.relu(scalarDiff)
    #print('printing1')
    #print(torch.sum(scalarDiff))
    scalarError = scalarDiff*scalarDiff
    scalarError = scalarError[scalarError>0]
    scalarLoss = torch.sum(scalarError)
    eleNum = scalarError.shape[0]
    scalarLoss = scalarLoss/(1+eleNum)
    
    return scalarLoss


def intersectionLoss(points, grads, scalars, model, x_lim, y_lim, z_lim):
    #model.eval()
    gradNorms = (torch.norm(grads, dim=1)).unsqueeze(1)
    gradDirs = grads/(gradNorms+1e-8)
    gradDirs = gradDirs*0.1
    
    
    p0 = points.detach()
    
    p1 = p0+0.5*gradDirs
    p2 = p0+1*gradDirs    
    p3 = p0+7.5*gradDirs
    p4 = p0+3*gradDirs
    
    
    scalars = scalars#.detach()
    if(len(scalars.shape)<2):
        scalars = scalars.unsqueeze(1)
    
    scalars = scalars.repeat(3,1)
    
    new_points = torch.vstack((p1,p2,p4))
    outScalars = model(new_points)['scalars']    
    
    scalarDiff = scalars - outScalars
    #print(scalarDiff.shape)
    scalarDiff = torch.relu(scalarDiff)
    #print('printing1')
    #print(torch.sum(scalarDiff))
    scalarError = scalarDiff*scalarDiff
    
    insideMask = getPointInsideDomainMask(new_points, x_lim, y_lim, z_lim)
    scalarError = scalarError[insideMask]
    scalarError = scalarError[scalarError>0]
    scalarLoss = torch.sum(scalarError)
    eleNum = scalarError.shape[0]
    scalarLoss = scalarLoss/(1+eleNum)

    
    p1 = p0-0.5*gradDirs
    p2 = p0-1*gradDirs    
    p3 = p0-3*gradDirs
    p4 = p0-10*gradDirs
     
     
    new_points = torch.vstack((p1,p2,p3))
    outScalars = model(new_points)['scalars']    
     
    scalarDiff = outScalars - scalars
    
    scalarDiff = torch.relu(scalarDiff)
    #print('printing2')
    #print(torch.sum(scalarDiff))
    
    scalarError = scalarDiff*scalarDiff
    
    insideMask = getPointInsideDomainMask(new_points)
    scalarError = scalarError[insideMask]
    scalarError = scalarError[scalarError>0]
    scalarLoss2 = torch.sum(scalarError)
    eleNum = scalarError.shape[0]
    scalarLoss2 = scalarLoss2/(1+eleNum)
    
    
    model.train()
    
    return scalarLoss2+scalarLoss

def intersectionLoss2(points, grads, scalars, model, clearance, x_lim, y_lim, z_lim):
    #model.eval()
    gradNorms = (torch.norm(grads, dim=1)).unsqueeze(1)
    gradDirs = grads/(gradNorms+1e-8)
    gradDirs = gradDirs*0.1
    #gradDirs = gradDirs.detach()
    
    p0 = points.detach()

    p1 = p0+0.5*gradDirs
    p2 = p0+0.8*gradDirs    
    p3 = p0+4.0*gradDirs
    p4 = p0+0.3*gradDirs
    
    
    #scalars = scalars.detach()
    if(len(scalars.shape)<2):
        scalars = scalars.unsqueeze(1)
    
    scalars = scalars.repeat(4,1)
    gradNormsRepeat = gradNorms.repeat(4,1).detach()
    print(gradNormsRepeat.shape)
    
    new_points = torch.vstack((p1,p2,p3,p4))
    #new_points.requires_grad = True
    outPuts = model(new_points)
    outScalars = outPuts['scalars']
    outGrads = outPuts['grads']
    outNorms = torch.norm(outGrads, dim=1).detach()    
    
    sample_heights = torch.tensor([0.05,0.08,0.40,0.03], dtype= torch.float32).reshape(4,1)
    
    sample_heights = sample_heights.repeat(1,int((new_points.shape[0])/4))
    sample_heights = sample_heights.reshape(scalars.shape)
    sample_heights = sample_heights.to(device)
    #print(sample_heights)
    clearance = defineCone(sample_heights)
    print(clearance.shape)
    print(scalars.shape)
    scalarDiff = scalars - outScalars + outNorms.unsqueeze(1)*clearance.detach()
    print(scalarDiff.shape)
    scalarDiff = torch.relu(10*scalarDiff)
    #print('printing1')
    #print(torch.sum(scalarDiff))
    scalarError = scalarDiff*scalarDiff
    scalarsMask = scalarError>0
   
    
    #scalarLoss = torch.mean(scalarDiff)
    insideMask = getPointInsideDomainMask(new_points, x_lim, y_lim, z_lim)

    scalarsMask = scalarsMask*insideMask.unsqueeze(1)
    
    collidedPoints = new_points[scalarsMask.flatten()]
    
    scalarError = scalarError[insideMask]
    
    #scalarLoss = torch.mean(scalarError)

    scalarsMask = scalarsMask.reshape(-1,int((new_points.shape[0])/4))
    scalarsMask = scalarsMask.T
    selectedScalars = torch.sum(scalarsMask, dim=1)
    
    scalarsMask = selectedScalars>0
    
    scalarError = scalarError[scalarError>0]
    
    scalarLoss = torch.sum(scalarError)
    eleNum = scalarError.shape[0]
    scalarLoss = scalarLoss/(1+eleNum)

    '''
    p1 = p0-0.5*gradDirs
    p2 = p0-1*gradDirs    
    p3 = p0-2*gradDirs
    p4 = p0-3*gradDirs
     
     
    new_points = torch.vstack((p1,p3,p4))
    outScalars = model(new_points)['scalars']    
     
    scalarDiff = outScalars - scalars
    
    scalarDiff = torch.relu(scalarDiff)
    #print('printing2')
    #print(torch.sum(scalarDiff))
    
    scalarError = scalarDiff*scalarDiff
    scalarError = scalarError[scalarError>0]
    scalarLoss2 = torch.sum(scalarError)
    eleNum = scalarError.shape[0]
    scalarLoss2 = scalarLoss2/(1+eleNum)
    '''
    
    model.train()
    
    return {'loss':scalarLoss, 'mask':scalarsMask, 'colPoints':collidedPoints}#+scalarLoss2





##########################################################################

def dispComplianceHisto(points, targets, model, targetType = 0): 
    '''

    Parameters
    ----------
    points : TYPE
        DESCRIPTION.
    targets : TYPE
        DESCRIPTION.
    targetType : int, optional
        targetType to determine the type of metric to evaluate- 
        grads(0) or scalars(1) . The default is 0.

    Returns
    -------
    None.

    '''
    model.eval()
    out = model(points)
    if targetType==0:
        grads = out['grads'].detach()
        gradNorm = torch.norm(grads, dim=1).unsqueeze(1)
        grads = grads/gradNorm
        targets = targets.detach()
        dotProd = grads*targets
        dotProd = torch.sum(dotProd, dim=1)
        angles = torch.acos(dotProd).to('cpu').numpy().flatten()
        angles = angles*(180.00/3.1457)
        
        bin_edges = np.linspace(70,110,200)  # 6 bins with edges at 0, 1, 2, 3, 4, 5, 6

        # Create a histogram with custom bin limits
        plt.hist(angles, bins=bin_edges, edgecolor='black')
        plt.show()
    
    return angles


def dispComplianceHistoTP(points, targets, model, model2, targetType = 0): 
    '''

    Parameters
    ----------
    points : TYPE
        DESCRIPTION.
    targets : TYPE
        DESCRIPTION.
    targetType : int, optional
        targetType to determine the type of metric to evaluate- 
        grads(0) or scalars(1) . The default is 0.

    Returns
    -------
    None.

    '''
    model.eval()
    model2.eval()
    out = model(points)
    out2 = model2(points)
    if targetType==0:
        grad1 = out['grads'].detach()
        grad2 = out2['grads'].detach()
        grads = torch.cross(grad1, grad2)
        gradNorm = torch.norm(grads, dim=1).unsqueeze(1)
        grads = grads/gradNorm
        targets = targets.detach()
        dotProd = grads*targets
        dotProd = torch.abs(torch.sum(dotProd, dim=1))
        angles = torch.acos(dotProd).to('cpu').numpy().flatten()
        angles = angles*(180.00/3.1457)
        
        bin_edges = np.linspace(0,180,1000)  # 6 bins with edges at 0, 1, 2, 3, 4, 5, 6

        # Create a histogram with custom bin limits
        plt.hist(angles, bins=bin_edges, edgecolor='black')
        plt.show()
        
        return angles



def detectEdgePoints(boundaryPoints, boundaryNormals, field1):
    fieldGrads = field1(boundaryPoints)['grads']
    dotProd = fieldGrads.detach()*boundaryPoints
    dotProd = torch.sum(dotProd, dim=1)
    
    dotProd = torch.abs(dotProd)
    boundaryMask = (dotProd.to('cpu').detach().numpy())<0.2
    
    edgePoints = boundaryPoints[boundaryMask]
    #edgePoints = edgePoints.detach()
    #edgePoints.requires_grad = True
    edgeNormals = boundaryNormals[boundaryMask]
    return {'points':edgePoints, 'normals':edgeNormals}
 


def supportLoss(surfaceNormals, surfaceGrads):
    gradNorm = torch.norm(surfaceGrads, dim=1).unsqueeze(1)
    surfaceGrads = surfaceGrads/(gradNorm+1e-10)
    dotProd = surfaceNormals*surfaceGrads
    dotProd = torch.sum(dotProd, dim=1)
    
    
    #gradNorm = torch.norm(surfaceGrads, dim=1)
    #assert(dotProd.shape == gradNorm.shape)
    supportError = -dotProd+np.cos(133.00*3.1457/180.00)
    supportMask = torch.relu(supportError)>0.0
    #print(supportError.max())
    supportError = torch.relu(10*supportError)
    #supportError = torch.sigmoid(30*supportError)
    supportMask = supportError>0.0
    #supportErrorSelects = supportError[supportError>0]
    #supportErrorNum = supportErrorSelects.shape[0]
    supportLoss = torch.mean(supportError*supportError)#/(supportErrorNum+1)
    return {'loss':supportLoss,'mask':supportMask}




def computeGaussianCurvature(dx2,dy2,dz2,grads):
    fxx = dx2[:,0]
    fxy = dx2[:,1]
    fxz = dx2[:,2]
    fyy = dy2[:,1]
    fyz = dy2[:,2]
    fzz = dz2[:,2]
    
    
    
    h11 = fyy*fzz-fyz*fyz
    h12 = fyz*fxz-fxy*fzz
    h13 = fxy*fyz-fyy*fxz
    h22 = fxx*fzz-fxz*fxz
    h23 = fxy*fxz-fxx*fyz
    h33 = fxx*fyy-fxy*fxy
    
    fx = grads[:,0]#.detach()
    fy = grads[:,1]#.detach()
    fz = grads[:,2]#.detach()
    
    norm_gradF = torch.norm(grads,dim=1)#.detach()
    #print(norm_gradF)    
    
    Kg_num = fx*fx*h11+fy*fy*h22+fz*fz*h33
    #print(Kg_num.shape)
    Kg_num = Kg_num + 2*h12*fx*fy+2*h13*fx*fz+2*h23*fy*fz
    Kg_den = norm_gradF*norm_gradF*norm_gradF*norm_gradF+1e-10
    #print(Kg_den.shape)
    
    
    
    Kg = Kg_num/Kg_den
    
    Kg = Kg.unsqueeze(1)
    
    return Kg


def computeMeanCurvature(dx2,dy2,dz2,grads):
    
    fxx = dx2[:,0]
    fxy = dx2[:,1]
    fxz = dx2[:,2]
    fyy = dy2[:,1]
    fyz = dy2[:,2]
    fzz = dz2[:,2]
    #
    #
    #
    h11 = fxx
    h12 = fxy
    h13 = fxz
    h22 = fyy
    h23 = fyz
    h33 = fzz
    #
    fx = grads[:,0]#.detach()
    fy = grads[:,1]#.detach()
    fz = grads[:,2]#.detach()
    #
    norm_gradF = torch.norm(grads,dim=1)#.detach()
    #
    Km_num = fx*fx*h11+fy*fy*h22+fz*fz*h33
    #print(Km_num.shape)
    Km_num = Km_num + 2*h12*fx*fy+2*h13*fx*fz+2*h23*fy*fz
    trace_h = h11+h22+h33
    Km_num = Km_num - norm_gradF*norm_gradF*trace_h
    Km_den = 2*norm_gradF*norm_gradF*norm_gradF+1e-10
    #print(Km_den.shape)
    #
    #
    Km = Km_num/Km_den
    #
    Km = Km.unsqueeze(1)
    #
    return Km


def computePrincipalCurvatures(dx2,dy2,dz2,grads):
    Kg = computeGaussianCurvature(dx2, dy2, dz2, grads)
    Km = computeMeanCurvature(dx2, dy2, dz2, grads)
    
    K1 = Km + torch.sqrt(Km*Km - Kg + 1e-8)
    K2 = Km - torch.sqrt(Km*Km - Kg + 1e-8)
    
    curvatures = torch.hstack((K1,K2))
    
    
    return curvatures


#################Circle/Sphere Class for test##############################

class sphere(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,input_coords):
        x= input_coords[:,0]
        y= input_coords[:,1]
        z= input_coords[:,2]
        
        out = torch.sqrt(x*x+y*y+z*z)
        out = out.unsqueeze(1)
        
        grads = torch.autograd.grad(out, input_coords, torch.ones_like(out), create_graph=True)[0]
        
        dx = grads[:,0]
        dy = grads[:,1]
        dz = grads[:,2]
        
        
        gradsx2 = torch.autograd.grad(dx,input_coords, torch.ones_like(dx), create_graph=True)[0]
        gradsy2 = torch.autograd.grad(dy,input_coords, torch.ones_like(dy), create_graph=True)[0]
        gradsz2 = torch.autograd.grad(dz,input_coords, torch.ones_like(dz), create_graph=True)[0]
        
        return {'scalars':out, 'grads':grads, 'HX2':gradsx2, 'HY2':gradsy2, 'HZ2':gradsz2}


class circle(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,input_coords):
        x= input_coords[:,0]
        y= input_coords[:,1]
        z= input_coords[:,2]
        
        out = torch.sqrt(x*x+z*z)
        out = out.unsqueeze(1)
        
        grads = torch.autograd.grad(out, input_coords, torch.ones_like(out), create_graph=True)[0]
        
        dx = grads[:,0]
        dy = grads[:,1]
        dz = grads[:,2]
        
        
        gradsx2 = torch.autograd.grad(dx,input_coords, torch.ones_like(dx), create_graph=True)[0]
        gradsy2 = torch.autograd.grad(dy,input_coords, torch.ones_like(dy), create_graph=True)[0]
        gradsz2 = torch.autograd.grad(dz,input_coords, torch.ones_like(dz), create_graph=True)[0]
        
        return {'scalars':out, 'grads':grads, 'HX2':gradsx2, 'HY2':gradsy2, 'HZ2':gradsz2}


####################################################################


def getTargetSphereGaussCurvature(points):
    rad = torch.norm(points, dim=1).unsqueeze(1)
    gaussCurv = 1/(rad*rad+1e-10)
    
    return gaussCurv
    
    
def computeAreaLoss2(scalars, grads, selectedIndex=None):
    
    selectedScalar = scalars.detach()[selectedIndex]
    gradMags = torch.norm(grads, dim=1).unsqueeze(1)
    gradMags = gradMags.repeat(1,1250)#.detach()
    
    beta = 1
    n = len(scalars)
    scalarMax = (1/beta)*torch.log((1/n)*torch.sum(torch.exp(beta*scalars)))
    scalarMin = -(1/beta)*torch.log((1/n)*torch.sum(torch.exp(-beta*scalars)))
    
    scalarMax = torch.max(scalars).detach()
    scalarMin = torch.min(scalars).detach()
    
    #scalars = (scalars-scalarMin)/(scalarMax-scalarMin)
    scalars = scalars.repeat(1,1250)
    print(scalars.shape)
    scalarsL = scalarMin + 0.05*(scalarMax-scalarMin)
    scalarsH = scalarMin + 0.95*(scalarMax-scalarMin)
    means = torch.linspace(scalarsL,scalarsH,1250).to(device)
    #print(selectedScalar)
    #means = torch.linspace(selectedScalar.to('cpu').numpy()[0][0]+0.01,selectedScalar.to('cpu').numpy()[0][0]+0.06,250).to(device)
    #print(means)
    means = means.reshape(1,-1)
    print(means.shape)
    #print(means)
    diff = scalars - means
    
    
    diff2 = -100*diff*diff
    #print(diff2.shape)
    div = (scalarMax-scalarMin)/100
    
    gaussians = torch.exp(diff2)
    mask = gaussians>0.95
    print(mask.shape)
    gaussians = gaussians*gradMags
    
    
    bin_sums = torch.sum(gaussians, dim=0)
    areas,indices = torch.topk(bin_sums, k=20, largest=False)
    min_index = indices[-1]
    min_val = means[0,min_index.detach()]
    min_area = areas[-1]
    
    
    areas,indices = torch.topk(bin_sums, k=20, largest=True)
    max_area = areas[0] 
    
    
    scalarsL = min_val - 0.01
    scalarsH = min_val + 0.01
    means = torch.linspace(scalarsL,scalarsH,1250).to(device)
    #print(selectedScalar)
    #means = torch.linspace(selectedScalar.to('cpu').numpy()[0][0]+0.01,selectedScalar.to('cpu').numpy()[0][0]+0.06,250).to(device)
    #print(means)
    means = means.reshape(1,-1)
    print(means.shape)
    #print(means)
    diff = scalars - means
    
    
    diff2 = -100*diff*diff
    #print(diff2.shape)
    div = (scalarMax-scalarMin)/100
    
    gaussians = torch.exp(diff2)
    mask = gaussians>0.95
    print(mask.shape)
    gaussians = gaussians*gradMags
    
    
    bin_sums = torch.sum(gaussians, dim=0)
    target = torch.min(1.1*min_area,0.5*max_area)
    bin_error = nn.functional.relu(target - bin_sums)
    
    bin_loss = torch.mean(bin_error)
    
    return bin_loss   

    
    
    
    
def computeAreaLoss(scalars, grads, selectedIndex=None):
    
    selectedScalar = scalars.detach()[selectedIndex]
    gradMags = torch.norm(grads, dim=1).unsqueeze(1)
    gradMags = gradMags.repeat(1,1250)#.detach()
    
    beta = 1
    n = len(scalars)
    scalarMax = (1/beta)*torch.log((1/n)*torch.sum(torch.exp(beta*scalars)))
    scalarMin = -(1/beta)*torch.log((1/n)*torch.sum(torch.exp(-beta*scalars)))
    
    scalarMax = torch.max(scalars).detach()
    scalarMin = torch.min(scalars).detach()
    
    #scalars = (scalars-scalarMin)/(scalarMax-scalarMin)
    scalars = scalars.repeat(1,1250)
    print(scalars.shape)
    scalarsL = scalarMin + 0.45*(scalarMax-scalarMin)
    scalarsH = scalarMin + 0.55*(scalarMax-scalarMin)
    means = torch.linspace(scalarsL,scalarsH,1250).to(device)
    #print(selectedScalar)
    #means = torch.linspace(selectedScalar.to('cpu').numpy()[0][0]+0.01,selectedScalar.to('cpu').numpy()[0][0]+0.06,250).to(device)
    #print(means)
    means = means.reshape(1,-1)
    print(means.shape)
    #print(means)
    diff = scalars - means
    
    
    diff2 = -100*diff*diff
    #print(diff2.shape)
    div = (scalarMax-scalarMin)/100
    
    gaussians = torch.exp(diff2)
    mask = gaussians>0.95
    print(mask.shape)
    gaussians = gaussians*gradMags
    
    print(gaussians.shape)
    #print(torch.max(gaussians, dim=0))
    bin_sums = torch.sum(gaussians, dim=0)
    mean= torch.mean(bin_sums).detach()
    bin_error = nn.functional.relu(17000 - bin_sums)
    bin_error = bin_error
    print(torch.min(bin_sums))
    print(torch.max(bin_sums))
    #bin_error = torch.abs(6800 - bin_sums)
    #print(bin_sums)
    #print(bin_error)
    print(bin_error.shape)
    bin_loss = torch.mean(bin_error)
    
    return bin_loss
    
    



def computeLengthLoss(scalars, scalars2, grads, grads2):
    gradMags = torch.norm(grads, dim=1).unsqueeze(1)
    gradMags = gradMags.repeat(1,50)#.detach()
    
    n = len(scalars)

    scalarMax = torch.max(scalars).detach()
    scalarMin = torch.min(scalars).detach()
    
    scalars = (scalars-scalarMin)/(scalarMax-scalarMin)
    scalars = scalars.repeat(1,50)
    means = torch.linspace(0.4, 0.6,50).to(device)
    means = means.reshape(1,-1)
    diff = scalars - means
    
    
    diff_2 = -200*diff*diff
    gaussians = torch.exp(diff_2)
    gaussians_pre = gaussians.unsqueeze(0).repeat(330,1,1)
    gaussians = gaussians*gradMags
    gaussians = gaussians.unsqueeze(0).repeat(330,1,1)
    
    
    projectedMags = torch.sum(grads*grads2, dim=1).unsqueeze(1)
    projectedGrads = grads2 - projectedMags*grads
    
    gradMags2 = torch.norm(projectedGrads, dim=1).unsqueeze(1)
    gradMags2 = gradMags2.repeat(1,330)#.detach()

    n2 = len(scalars2)
    
    scalarMax2 = torch.max(scalars2).detach()
    scalarMin2 = torch.min(scalars2).detach()
    #print(scalarMin2)
    #print(scalarMax2)
    #scalars2 = (scalars2-scalarMin2)/(scalarMax2-scalarMin2)
    scalars2_lowEnd = scalarMin2 + 0.1
    scalars2_upperEnd = scalarMax2 - 0.1
    scalars2 = scalars2.repeat(1,330)
    means2 = torch.linspace(scalars2_lowEnd,scalars2_upperEnd,330).to(device)
    means2 = means2.reshape(1,-1)
    diff2 = scalars2 - means2
    
    
    diff2_2 = -50*diff2*diff2
    gaussians2 = torch.exp(diff2_2)
    #print(torch.max(gaussians2))
    mask = gaussians2<0.6


    gaussians2_pre = gaussians2.unsqueeze(0).repeat(50,1,1).permute(2,1,0)   
    gaussians2 = gaussians2*gradMags2
    gaussians2 = gaussians2.unsqueeze(0).repeat(50,1,1)
    gaussians2 = gaussians2.permute(2,1,0)
    maxval = torch.max(gaussians2)
    
    
    combinedGaussians_pre = gaussians_pre*gaussians2_pre
    combinedGaussians = gaussians*gaussians2
    selectedMask = combinedGaussians>0.88
    selectedMask2 = selectedMask
    selectedMask = selectedMask[16,:,35]
    
    
    maxes = torch.max(combinedGaussians_pre, dim=1)[0]
    print(maxes[1,20])
    influence_mask = maxes>0.88
    
    #print(maxes)
    
    binGaussians = torch.sum(combinedGaussians, dim=1)
    layer_influence_numbers = torch.sum(influence_mask, dim=0)
    layer_sums = torch.sum(influence_mask*binGaussians, dim=0)
    layer_means = layer_sums/layer_influence_numbers
    layer_means = layer_means.reshape(1,binGaussians.shape[1])
    
    #print(influence_mask.shape)
    #print(layer_influence_numbers)
    
    print((binGaussians == torch.max(binGaussians[influence_mask])).nonzero())
    print(torch.min(binGaussians[influence_mask]))
    print(torch.max(binGaussians[influence_mask]))
    #print(layer_means.shape)
    
    #print(binGaussians[:,49])
    
    #bin_error = 0.50*layer_means - binGaussians
    bin_error = 400.00 - binGaussians
    bin_error = influence_mask*bin_error
    bin_error = nn.functional.relu(bin_error)
    lengthLoss = torch.mean(bin_error)
    print(binGaussians[1,20])
    print(influence_mask[1,20])
    
    print('lengths')
    print(torch.min(binGaussians[influence_mask]))
    print(torch.max(binGaussians[influence_mask]))
    
    return {'loss':lengthLoss,'mask':selectedMask}    
    


def lengthBoundaryCrossing(surfPoints, field1, field2):
    
    out1 = field1(surfPoints)
    scalars = out1['scalars'].detach()
    grads = out1['grads'].detach()
    
    out2 = field2(surfPoints)
    scalars2 = out2['scalars']
    grads2 = out2['grads']
    
    
    gradMags = torch.norm(grads, dim=1).unsqueeze(1)
    gradMags = gradMags.repeat(1,50).detach()
    
    n = len(scalars)

    scalarMax = torch.max(scalars).detach()
    scalarMin = torch.min(scalars).detach()
    
    scalars = (scalars-scalarMin)/(scalarMax-scalarMin)
    scalars = scalars.repeat(1,50)
    means = torch.linspace(0.2, 0.8,50).to(device)
    means = means.reshape(1,-1)
    diff = scalars - means
    
    
    diff_2 = -1000*diff*diff
    gaussians = torch.exp(diff_2)
    gaussians_pre = gaussians.unsqueeze(0).repeat(20,1,1)
    gaussians = gaussians*gradMags
    gaussians = gaussians.unsqueeze(0).repeat(20,1,1)
    
    
    gradMags2 = torch.norm(grads2, dim=1).unsqueeze(1)
    gradMags2 = gradMags2.repeat(1,20).detach()

    n2 = len(scalars2)
    
    scalarMax2 = torch.max(scalars2).detach()
    scalarMin2 = torch.min(scalars2).detach()
    #print(scalarMin2)
    #print(scalarMax2)
    scalars2 = (scalars2-scalarMin2)/(scalarMax2-scalarMin2)
    scalars2 = scalars2.repeat(1,20)
    means2 = torch.linspace(0.2, 0.8,20).to(device)
    means2 = means2.reshape(1,-1)
    diff2 = scalars2 - means2
    
    
    diff2_2 = -200*diff2*diff2
    gaussians2 = torch.exp(diff2_2)
    #print(torch.max(gaussians2))
    mask = gaussians2<0.6


    gaussians2_pre = gaussians2.unsqueeze(0).repeat(50,1,1).permute(2,1,0)   
    gaussians2 = gaussians2*gradMags2
    gaussians2 = gaussians2.unsqueeze(0).repeat(50,1,1)
    gaussians2 = gaussians2.permute(2,1,0)
    maxval = torch.max(gaussians2)
    #print(maxval)
    
    combinedGaussians_pre = gaussians_pre*gaussians2_pre
    combinedGaussians = gaussians*gaussians2
    
    maxes = torch.max(combinedGaussians_pre, dim=1)[0]
    influence_mask = maxes>0.95
    
    selectedMask = combinedGaussians>0.85
    sums = torch.sum(combinedGaussians*selectedMask, dim = 1)
    
    
    sumsError = nn.functional.relu(sums - 8.0) 
    sumsLoss = sumsError
    
    loss = torch.mean(sumsLoss*influence_mask)
    
    print(sums[10,10])
    
    
    selectedMask = selectedMask[10,:,10]
    return {'mask':selectedMask,'loss':loss}
        
    
     
    

def getCollisionTestField(points):
    x = points[:,0]
    absX = torch.abs(x)
    
    x_dirs = x/absX
    x_dirs = x_dirs.unsqueeze(1)
    zeroStack = torch.zeros_like(x_dirs)
    oneStack = torch.ones_like(x_dirs)
    normals = torch.hstack((-x_dirs, zeroStack, oneStack))
    normalsNorm = torch.norm(normals, dim=1).unsqueeze(1)
    normals = normals/normalsNorm
    
    return normals.detach()



def getMaxCurvature(points, grads, d2X, g2Y, d2Z):
    
    
    
    pass

#########################################################################


def defineCone(sample_height):
    theta = 3.1457/2.5
    sin_t = np.sin(theta)
    max_val = 0.15*sin_t
    return_vals = sample_height*sin_t
    return_vals = torch.clamp(return_vals, max = max_val)
    return return_vals

def getPointInsideDomainMask(points,x_lim =1.0, y_lim=1.0, z_lim=1.0):
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    
    check1 = abs(x)<x_lim
    check2 = abs(y)<y_lim
    check3 = abs(z)<z_lim
    
    mask = check1 & check2 & check3
    
    return mask


def getTPcurvatureLoss(field1Grads, field2Grads, field2_d2X, field2_d2Y, field2_d2Z, scale):
    norm_grads2 = torch.norm(field2Grads, dim=1)#.unsqueeze(0)
    
    h11 = field2_d2X[:,0]
    h22 = field2_d2Y[:,1]
    h33 = field2_d2Z[:,2]
    h12 = field2_d2X[:,1]
    h13 = field2_d2X[:,2]
    h23 = field2_d2Y[:,2]
    
    
    field1Grads = field1Grads/(torch.norm(field1Grads, dim=1, keepdim=True)+1e-10)
    field2Grads = field2Grads/(torch.norm(field2Grads, dim=1, keepdim=True)+1e-10)
    tangentDirs = torch.cross(field1Grads, field2Grads)
    tangentNorms = torch.norm(tangentDirs, dim=1).unsqueeze(1)
    tangentDirs = tangentDirs/(tangentNorms+1e-10)
    tx = tangentDirs[:,0]
    ty = tangentDirs[:,1]
    tz = tangentDirs[:,2]
    
    curvature = tx*tx*h11 + ty*ty*h22 + tz*tz*h33
    curvature = curvature + 2*tx*ty*h12 + 2*tx*tz*h13 + 2*ty*tz*h23
    #print(curvature.shape)
    
    curvature = curvature/(norm_grads2+1e-10)
    #print(curvature.shape)
    #curvature = curvature/scale
    
    #print(curvature)
    #print(torch.max(curvature))
    
    return curvature




def platformCollisionScalarError(platformBase, platfromNormal, scalarField, points, grads):
    '''
    Parameters
    ----------
    platformBase : TYPE
        DESCRIPTION.
    platfromNormal : TYPE
        DESCRIPTION.
    scalarField : TYPE
        DESCRIPTION.
    points : TYPE
        DESCRIPTION.
    grads : TYPE
        Return Loss based on distance to platform 

    Returns
    -------
    None.

    '''
    pass

def platformCollisionGradientError(platformNormal, scalarField, points, grads):
    '''
    Parameters
    ----------
    platformBase : TYPE
        DESCRIPTION.
    platfromNormal : TYPE
        DESCRIPTION.
    scalarField : TYPE
        DESCRIPTION.
    points : TYPE
        DESCRIPTION.
    grads : TYPE
        Return Loss based on gradients of points and directions to platform 

    Returns
    -------
    None.

    '''
    gradNorms = torch.norm(grads, dim=1).unsqueeze(1)
    gradDirs = grads/gradNorms
    
    platformNormal = platformNormal.view(1,3)
    dotProd = torch.sum(platformNormal*gradDirs, dim=1)
    platformColError = torch.relu(-10*dotProd)
    platformLoss = torch.mean(platformColError*platformColError)
    
    return platformLoss
 
    
def get_platformData(points, scalars, grads):
    minScalar = torch.min(scalars)
    maxScalar = torch.max(scalars)

    thresholdScalar = minScalar + 3e-1*(maxScalar - minScalar)
    baseSelectMask = scalars < thresholdScalar

    pickedPoints = input_points[baseSelectMask.flatten()]*rangeVals + midVals
    pickedGrads = grads[baseSelectMask.flatten()]
    platformDir = torch.mean(pickedGrads, dim=0)
    platformDir = platformDir/torch.norm(platformDir)
    
    dirDotProd = torch.sum(pickedPoints*(platformDir.view(1,3)), dim=1)
    platformBasePointIndex = torch.nonzero(dirDotProd == torch.min(dirDotProd), as_tuple=True)[0].item()
    platformBasePoint = pickedPoints[platformBasePointIndex]
    platformBasePoint = platformBasePoint - 10*platformDir
    
    return {'point':platformBasePoint.detach(), 'dir':platformDir.detach()}








class platformModel(nn.Module):
    def __init__(self, device='cuda', scale=1.0):
        super().__init__()
        self.device = device
        self.platformBase = nn.Parameter(torch.tensor([[0,0,0]], dtype=torch.float32, device=self.device))
        self.platformDir = nn.Parameter(torch.tensor([[0,1,0]], dtype=torch.float32, device=self.device))
        self.selectedPoints = None
        self.targetDirs = None
        self.dispDist = 5.00/scale
    
    def selectPoints(self, surfacePoints, surfaceGrads, surfaceNormals):
        ########
        #ideally should only include the surface point that are also part of the convex hull!!!
        ########
        
        gradNorm = torch.norm(surfaceGrads, dim=1).unsqueeze(1)
        surfaceGrads = surfaceGrads/(gradNorm+1e-10)
        dotProd = surfaceNormals*surfaceGrads
        dotProd = torch.sum(dotProd, dim=1)

        supportError = -dotProd+np.cos(137.00*3.1457/180.00)
        supportMask = torch.relu(supportError)>0.0
        
        self.selectedPoints = surfacePoints[supportMask].detach()
        self.selectedPoints.requires_grad = True
        self.targetDirs = surfaceGrads[supportMask].detach()
        
    
    def getPlatformPosLoss1(self, surfacePoints, surfaceGrads, surfaceNormals):
        
        #select surface points that need support
        self.selectPoints(surfacePoints, surfaceGrads, surfaceNormals)
        
        palformDirNorm = torch.norm(self.platformDir)+1e-10
        
        #calculate the orientaton loss so that the difference between the gradient aat those points
        #and surface normal is minimized
        if(self.targetDirs is not None):
            dirError = torch.sum(self.targetDirs*self.platformDir, dim=1)
            dirError = torch.relu(-dirError)
            dirLoss = torch.mean(dirError*dirError)
        else:
            dirLoss = 0
            
        
        
        #calculate the loss so that all surface points are above the plane (+d distance)
        #along the surface normal
        if(self.selectedPoints is not None):
            dispVector = surfacePoints.detach() - self.platformBase
            disps = torch.sum(dispVector*self.platformDir, dim=1)
            disps = self.dispDist - disps
            dispError = torch.relu(disps)
            errorMask = dispError>0
            dispLoss = torch.sum(dispError*dispError)/torch.sum(errorMask)
        else:
            dispLoss = 0
            
            
        #calculate the loss so that all surface points needing support are as close as possible to the plane 
        if(self.selectedPoints is not None):
            dispVector = self.selectedPoints.detach() - self.platformBase
            disps = torch.sum(dispVector*self.platformDir, dim=1)
            disps = disps - 0.05
            dispError2 = torch.relu(disps)
            dispLoss2 = torch.mean(dispError2*dispError2)
        else:
            dispLoss2 = 0
        
        
        totalPlatformLoss = dirLoss + dispLoss + 0.05*dispLoss2
        return totalPlatformLoss
    
    
    def getPlatformPosLoss(self, surfacePoints, surfaceGrads, surfaceNormals):
        
        #select surface points that need support
        self.selectPoints(surfacePoints, surfaceGrads, surfaceNormals)
        
        palformDirNorm = torch.norm(self.platformDir)+1e-10

        
        #calculate the orientaton loss so that the difference between the gradient aat those points
        #and surface normal is minimized
        if(self.targetDirs is not None):
            dirError = self.targetDirs - (self.platformDir/palformDirNorm)
            dirLoss = torch.mean(dirError*dirError)
        else:
            dirLoss = 0
            
        
        
        #calculate the loss so that all surface points are above the plane (+d distance)
        #along the surface normal
        if(self.selectedPoints is not None):
            dispVector = surfacePoints.detach() - self.platformBase
            disps = torch.sum(dispVector*self.platformDir/palformDirNorm, dim=1)
            disps = self.dispDist - disps
            dispError = torch.relu(disps)
            errorMask = dispError>0
            dispLoss = torch.mean(dispError[errorMask]*dispError[errorMask])
            if(torch.sum(errorMask) < 1):
                dispLoss = 0
        else:
            dispLoss = 0
            
            
        #calculate the loss so that all surface points needing support are as close as possible to the plane 
        if(self.selectedPoints is not None):
            dispVector = self.selectedPoints.detach() - self.platformBase
            disps = torch.sum(dispVector*self.platformDir/palformDirNorm, dim=1)
            disps = disps - 0.05
            dispError2 = torch.relu(disps)
            dispLoss2 = torch.mean(dispError2*dispError2)
        else:
            dispLoss2 = 0

        totalPlatformLoss = dirLoss + dispLoss + 0.05*dispLoss2
        return totalPlatformLoss
        
        
    def getPlatfromPosLossDetached(self, surfacePoints, surfaceGrads):
        pass
    
    
    def getPlatformLoss_moveField(self, points, grads):
        pass
    
    def getPlatformLoss_movePlatfrom(self, points, grads):
        pass
    
    def getPlatformLoss_moveBoth(self, points, grads):
        pass


    
'''
Main Block
'''


################################### MAIN BLOCK ###############################################################

rand_indices = torch.randperm(maskedGrid.points.shape[0])
#print(maskedGrid.points.shape[0])
input_points = torch.tensor(maskedGrid.points[rand_indices][0:100000], dtype = torch.float32, device = device, requires_grad=True)
#input_points = torch.tensor(maskedGrid.points[rand_indices], dtype = torch.float32, device = device, requires_grad=True)


min_val = np.min([x_min, y_min, z_min])
max_val = np.max([x_max, y_max, z_max])
minVals = torch.tensor([x_min, y_min, z_min], dtype=torch.float32, device = device)
maxVals = torch.tensor([x_max, y_max, z_max], dtype=torch.float32, device = device)
midVals = 0.5*(minVals+maxVals)


max_range = np.max([x_max-x_min,y_max-y_min, z_max-z_min])
rangeVals = 0.5*torch.tensor([max_range,max_range,max_range], dtype=torch.float32, device=device)
input_points = (input_points - midVals)/rangeVals
x_lim, y_lim,z_lim = torch.max(input_points, dim=0)[0]

#x_lim = torch.max(torch.tensor([0.5,x_lim]))
#y_lim = torch.max(torch.tensor([0.5,y_lim]))
#z_lim = torch.max(torch.tensor([0.5,z_lim]))

input_points = input_points.detach()
input_points.requires_grad = True



#print("shape......")
# print(sPoints.shape)
# sPoints_np = sPoints
# sPoints = torch.tensor(sPoints, dtype= torch.float32, device=device)
sPoints = (sPoints - midVals)/rangeVals
sPoints = sPoints.detach()
rand_indices = torch.randperm(sPoints.shape[0])
########
#sPoints = sPoints[rand_indices][0:150000]
#maxStressNormals = maxStressNormals[rand_indices][0:150000]
###
sPoints.requires_grad = True


domainPoints = (domainPoints - midVals)/rangeVals
domainPoints = domainPoints.detach()
domainPoints.requires_grad = True


boundaryPoints = (boundaryPoints - midVals)/rangeVals
boundaryPoints = boundaryPoints.detach()
boundaryPoints.requires_grad = True


basePoints = (basePoints - midVals)/rangeVals
basePoints = basePoints.detach()
basePoints.requires_grad = True


connect_boundaryPoints = (connect_boundaryPoints - midVals)/rangeVals
connect_boundaryPoints = connect_boundaryPoints.detach()
connect_boundaryPoints.requires_grad = True


connect_connectPoints = (connect_connectPoints - midVals)/rangeVals
connect_connectPoints = connect_connectPoints.detach()
connect_connectPoints.requires_grad = True


isSupportLoss = False
isStressLoss = True


targetCurv = getTargetSphereGaussCurvature(input_points).detach()

envModel = obstacleFunction()


tYmin = torch.min(input_points[:,1])
index = (input_points[:,1]==tYmin).nonzero()
selectedPoints = input_points[index[0]].detach().to('cpu').numpy

#scalarField = sphere()
#scalarField2 = circle()


radii = torch.sqrt(input_points[:,0]*input_points[:,0]+input_points[:,2]*input_points[:,2])

targetK1 = -(1/(radii+1e-10)).detach().unsqueeze(1)
targetK2 = -(1/(radii+1e-10)).detach().unsqueeze(1)
targetK2 = torch.zeros_like(targetK1)
targetK = torch.hstack((targetK1,targetK2))

print("---")
print(input_points.shape)
print(sPoints.shape)
print("---")


input_point_batches = int(input_points.shape[0]/800)+1
sPoints_batches = int(sPoints.shape[0]/3000)+1

input_point_batchSize = int(input_points.shape[0]/input_point_batches)+1
sPoints_batchSize = int(sPoints.shape[0]/sPoints_batches)+1

input_points_dataLoader = DataLoader(input_points, batch_size=input_point_batchSize,shuffle=True)
domainPoints_dataLoader = DataLoader(domainPoints, batch_size=5000,shuffle=True)

sPoints_dataset = TensorDataset(sPoints, maxStressNormals)
sPoints_dataLoader = DataLoader(sPoints_dataset, batch_size=sPoints_batchSize,shuffle=True)


basePoint_dataLoader = DataLoader(basePoints, batch_size=20000,shuffle=True)

boundaryPoints_dataset = TensorDataset(boundaryPoints, boundaryNormals)
boundaryPoints_dataLoader = DataLoader(boundaryPoints_dataset, batch_size=5000,shuffle=True)

connectedPoints_dataset = TensorDataset(connect_boundaryPoints, connect_connectPoints, connect_boundaryNormals)
connectedPoints_dataLoader = DataLoader(connectedPoints_dataset, batch_size=30000,shuffle=True)



print(len(input_points_dataLoader))
print(len(sPoints_dataLoader))
print(len(domainPoints_dataLoader))

if(len(input_points_dataLoader)>len(sPoints_dataLoader)):
    sPoints_dataLoader = cycle(sPoints_dataLoader)
    domainPoints_dataLoader = cycle(domainPoints_dataLoader)
    basePoint_dataLoader = cycle(basePoint_dataLoader)
elif(len(sPoints_dataLoader)>len(input_points_dataLoader)):
    input_points_dataLoader = cycle(input_points_dataLoader)
    domainPoints_dataLoader = cycle(domainPoints_dataLoader)
    basePoint_dataLoader = cycle(basePoint_dataLoader)

# if(len(input_points_dataLoader)>len(sPoints_dataLoader)):
#     sPoints_dataLoader = cycle(sPoints_dataLoader)
#     domainPoints_dataLoader = cycle(domainPoints_dataLoader)

# elif(len(sPoints_dataLoader)>len(input_points_dataLoader)):
#     input_points_dataLoader = cycle(input_points_dataLoader)
#     domainPoints_dataLoader = cycle(domainPoints_dataLoader)



class combinedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = SirenNet(3, 128, 1, 10, w0_initial=20.0, w0 = 20.0)
        self.net2 = SirenNet(3, 128, 1, 10, w0_initial=20.0, w0 = 20.0)
        self.thetaNet = SirenNet(3, 128, 1, 2, w0_initial=20.0, w0 = 20.0)
        
    def forward(self, inps):
        out1 = self.net1(inps)
        out2 = self.net2(inps)
        
        out = out1['scalars'] + out2['scalars']
        grads = out1['grads'] + out2['grads'] 
        gradsx2 = out1['HX2'] + out2['HX2'] 
        gradsy2 = out1['HY2'] + out2['HY2'] 
        gradsz2 = out1['HZ2'] + out2['HZ2'] 
        
        
        return {'scalars':out, 'grads':grads, 'HX2':gradsx2, 'HY2':gradsy2, 'HZ2':gradsz2}

    
    

class residualSiren(nn.Module):
    def __init__(self):
        super().__init__()
        self.net_init = SirenNet(3, 128, 128, 5, w0_initial=7.0, w0 = 7.0, final_activation=Sine(w0=7.0))
        self.net_mid = SirenNet(128, 128, 128, 5, w0_initial=7.0, w0 = 7.0, final_activation=Sine(w0=7.0))
        self.net_end = SirenNet(128, 128, 1, 5, w0_initial=7.0, w0 = 7.0)
    
    def forward(self, inps):
        out_init = self.net_init(inps, compute_grads = False)
        scalar_init = out_init['scalars']
        
        in_mid = scalar_init
        out_mid = self.net_mid(in_mid, compute_grads = False)
        scalar_mid = out_mid['scalars']
        
        in_end = 0.1*(scalar_init) + 0.9*(scalar_mid)
        out_end = self.net_end(in_end, compute_grads = False)
        out = out_end['scalars']
        
        
        grads = torch.autograd.grad(out, inps, torch.ones_like(out), create_graph=True)[0]
        dx = grads[:,0]
        dy = grads[:,1]
        dz = grads[:,2]
        
        
        gradsx2 = torch.autograd.grad(dx,inps, torch.ones_like(dx), create_graph=True)[0]
        gradsy2 = torch.autograd.grad(dy,inps, torch.ones_like(dy), create_graph=True)[0]
        gradsz2 = torch.autograd.grad(dz,inps, torch.ones_like(dz), create_graph=True)[0]
        
        
        
        return {'scalars':out, 'grads':grads, 'HX2':gradsx2, 'HY2':gradsy2, 'HZ2':gradsz2}
    

class residualSiren10(nn.Module):
    def __init__(self):
        super().__init__()
        self.net_init = SirenNet(3, 128, 128, 5, w0_initial=10.0, w0 = 10.0, final_activation=Sine(w0=7.0))
        self.net_mid = SirenNet(128, 128, 128, 5, w0_initial=7.0, w0 = 7.0, final_activation=Sine(w0=7.0))
        self.net_end = SirenNet(128, 128, 1, 5, w0_initial=7.0, w0 = 7.0)
    
    def forward(self, inps):
        out_init = self.net_init(inps, compute_grads = False)
        scalar_init = out_init['scalars']
        
        in_mid = scalar_init
        out_mid = self.net_mid(in_mid, compute_grads = False)
        scalar_mid = out_mid['scalars']
        
        in_end = 0.1*(scalar_init) + 0.9*(scalar_mid)
        out_end = self.net_end(in_end, compute_grads = False)
        out = out_end['scalars']
        
        
        grads = torch.autograd.grad(out, inps, torch.ones_like(out), create_graph=True)[0]
        dx = grads[:,0]
        dy = grads[:,1]
        dz = grads[:,2]
        
        
        gradsx2 = torch.autograd.grad(dx,inps, torch.ones_like(dx), create_graph=True)[0]
        gradsy2 = torch.autograd.grad(dy,inps, torch.ones_like(dy), create_graph=True)[0]
        gradsz2 = torch.autograd.grad(dz,inps, torch.ones_like(dz), create_graph=True)[0]
        
        
        
        return {'scalars':out, 'grads':grads, 'HX2':gradsx2, 'HY2':gradsy2, 'HZ2':gradsz2}

    
class residualSiren4(nn.Module):
    def __init__(self):
        super().__init__()
        self.net_init = SirenNet(3, 128, 128, 5, w0_initial=4.0, w0 = 4.0, final_activation=Sine(w0=4.0))
        self.net_mid = SirenNet(128, 128, 128, 5, w0_initial=4.0, w0 = 4.0, final_activation=Sine(w0=4.0))
        self.net_end = SirenNet(128, 128, 1, 5, w0_initial=4.0, w0 = 4.0)
    
    def forward(self, inps):
        out_init = self.net_init(inps, compute_grads = False)
        scalar_init = out_init['scalars']
        
        in_mid = scalar_init
        out_mid = self.net_mid(in_mid, compute_grads = False)
        scalar_mid = out_mid['scalars']
        
        in_end = 0.5*(scalar_init + scalar_mid)
        out_end = self.net_end(in_end, compute_grads = False)
        out = out_end['scalars']
        
        
        grads = torch.autograd.grad(out, inps, torch.ones_like(out), create_graph=True)[0]
        dx = grads[:,0]
        dy = grads[:,1]
        dz = grads[:,2]
        
        
        gradsx2 = torch.autograd.grad(dx,inps, torch.ones_like(dx), create_graph=True)[0]
        gradsy2 = torch.autograd.grad(dy,inps, torch.ones_like(dy), create_graph=True)[0]
        gradsz2 = torch.autograd.grad(dz,inps, torch.ones_like(dz), create_graph=True)[0]
        
        
        
        return {'scalars':out, 'grads':grads, 'HX2':gradsx2, 'HY2':gradsy2, 'HZ2':gradsz2}    
    

#scalarField = models.scalarNetwork().to(device)
#scalarField = SirenNet(3, 128, 1, 10, w0_initial=7.0, w0 = 7.0).to(device)
scalarField = SirenNet(3, 128, 1, 15, w0_initial=7.0, w0 = 7.0).to(device)
#scalarField = SirenNet(3, 128, 1, 6, w0_initial=5.0, w0 = 5.0).to(device)
#scalarField = scalarNet2().to(device)
#scalarField = DynamicLinearNet([3,256,256,256,256,256,256,256,256,1]).to(device)
#readPath = "parametersTest_batched_spiral_fish_10_128_15_15.pt"
#readPath = "parametersTest_batched_spiral_fish_mid23.pt"
readPath = "parametersTest_batched_TshapeBracketNew_deep4.pt"
#readPath = "parametersTest_batched_clip_mid8.pt"
#readPath = "parametersTest_batched_fertility_10_128_7_7.pt"
if model=='topopt':
    readPath = "parametersTest_batched_ncc3_mid.pt"
checkPoint = torch.load(readPath)
scalarField.load_state_dict(checkPoint['model_state_dict'])




scalarField2 = SirenNet(3, 128, 1, 15, w0_initial=7.0, w0 = 7.0).to(device)#combinedNet().to(device)#
readPath2 = "2parametersTest_batched_TshapeBracketNew_deep4.pt"
#readPath2 = "2parametersTest_batched_clip_mid8.pt"
checkPoint2 = torch.load(readPath2)
scalarField2.load_state_dict(checkPoint2['model_state_dict'])
platform = platformModel(scale=rangeVals[0])

# optimizer1 = torch.optim.SGD(list(scalarField.parameters()),lr = 5e-6, momentum=0.9)#0.03e-3 for support#0.5e-4 for area
# optimizer2 = torch.optim.SGD(list(scalarField2.parameters()),lr = 5e-6, momentum=0.9)#0.03e-3 for support#0.5e-4 for area
optimizer1 = torch.optim.Adam(list(scalarField.parameters()),lr = 5e-6)#0.03e-3 for support#0.5e-4 for area
optimizer2 = torch.optim.Adam(list(scalarField2.parameters()),lr = 5e-6)#0.03e-3 for support#0.5e-4 for area
optimizer3 = torch.optim.Adam(list(platform.parameters()),lr = 5e-2)#0.03e-3 for support#0.5e-4 for area




scheduler1 = lr_scheduler.StepLR(optimizer1, step_size = 500, gamma=0.4)
scheduler2 = lr_scheduler.StepLR(optimizer2, step_size = 500, gamma=0.4)

epochNum = 5000





colLossFun = collison_loss(20, 63, model_load_path='./clipSDF.pt')
colLossFun.init_tool_dense(rangeVals[0])


platfromDir = None
wt1 = 2e0
wt2 = 2e0

colLossOut = None


#y up 
class planarScalarField(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, points):
        scalars = (1*points[:,1]+0*points[:,0]+0*points[:,2]).unsqueeze(1)
        #scalars = scalars+scalars-scalars
        grads = torch.autograd.grad(scalars, points, torch.ones_like(scalars), create_graph = True)[0]
        
        dx  = grads[:,0]
        dy  = grads[:,1]
        dz  = grads[:,2]
        
        #print(grads)
        gradsx2 = None#torch.autograd.grad(dx, points, torch.ones_like(dx), create_graph = True)[0]
        gradsy2 = None#torch.autograd.grad(dy, points, torch.ones_like(dy), create_graph = True)[0]
        gradsz2 = None#torch.autograd.grad(dz, points, torch.ones_like(dz), create_graph = True)[0]

        return {'scalars':scalars, 'grads':grads, 'HX2':gradsx2, 'HY2':gradsy2, 'HZ2':gradsz2}



#y up 
class circleScalarField(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, points):
        x = points[:,0].unsqueeze(1)
        z = points[:,2].unsqueeze(1)
        
        scalars = (x*x+z*z).unsqueeze(1)
        
        grads = torch.autograd.grad(scalars, points, torch.ones_like(scalars), create_graph = True)[0]
        
        dx  = grads[:,0]
        dy  = grads[:,1]
        dz  = grads[:,2]
        
        
        gradsx2 = torch.autograd.grad(dx, points, torch.ones_like(dx), create_graph = True)[0]
        gradsy2 = torch.autograd.grad(dy, points, torch.ones_like(dy), create_graph = True)[0]
        gradsz2 = torch.autograd.grad(dz, points, torch.ones_like(dz), create_graph = True)[0]

        return {'scalars':scalars, 'grads':grads, 'HX2':gradsx2, 'HY2':gradsy2, 'HZ2':gradsz2}



# scalarField = planarScalarField()

# scalarField2 = circleScalarField()

# fullScalars = scalarField(input_points)['scalars'].detach()
# batchMax = torch.max(fullScalars).detach()
# batchMin = torch.min(fullScalars).detach()
# batchRange = batchMax - batchMin
# batchRangeLim = 0.5*batchRange

for epoch in range(1, epochNum):
    zippedSet = zip(input_points_dataLoader,basePoint_dataLoader,sPoints_dataLoader,domainPoints_dataLoader)
    batchLoss = 0
    iterCount = 0
    colRecord = 0
    curvRecord = 0
    fullScalars = scalarField(input_points)['scalars'].detach()
    batchMax = torch.max(fullScalars).detach()
    batchMin = torch.min(fullScalars).detach()
    batchRange = batchMax - batchMin
    batchRangeLim = 0.5*batchRange
    
    

    for batch_input_points, batch_basePoints, (batch_sPoints, batch_sDirns), batch_domainPoints in zippedSet:
        iterCount+=1
        # #batch_input_points.requires_grad = True
        # #batch_sPoints.requires_grad = True
        # #batch_domainPoints.requires_grad = True
        
        dispLoss = 0
        loss = 0
        
        # ########## Loss for InpPoints ################
        
        
        
        
        n = len(batch_input_points)
        inps = batch_input_points
        out = scalarField(inps)
        
        grads = out['grads']       
        
        gradNorm = torch.norm(grads, dim=1)
       
        
        gradNorm = torch.norm(grads, dim=1)
        gradNormGrad = torch.autograd.grad(gradNorm,inps, torch.ones_like(gradNorm), create_graph=True)[0]
            
        
        
        gradNormLoss = torch.mean(torch.abs(gradNormGrad*gradNormGrad))
        
        
        
        #prevent norm from getting too small:
        gradNormSquare = gradNorm*gradNorm
        smallGradLoss = torch.mean(torch.exp(-100*gradNormSquare))
        
        # targetGrad = torch.tensor([[0,0,1]], dtype= torch.float32, device=device)
        # targetGrad = targetGrad.repeat(grads.shape[0],1)
        # gradLoss = grads - targetGrad
        # gradLoss = torch.mean(gradLoss*gradLoss)
        

        
        if(True):#((epoch<10) or (epoch+1)%3==0):
            loss = 5e-1*gradNormLoss# + gradLoss # + 0*smallGradLoss 
            # field2Error = 1.0-out2['scalars']
            # field2Loss = torch.mean(torch.abs(field2Error))
            # loss += 1e-5*field2Loss
        
        
        dx2 = out['HX2'][:,0]
        dy2 = out['HY2'][:,1]
        dz2 = out['HZ2'][:,2]
        dxy = out['HX2'][:,1]
        dxz = out['HX2'][:,2]
        dyz = out['HY2'][:,2]
        

        lap = dx2 + dy2 + dz2
        lap = lap/(torch.norm(grads, dim=1)+1e-10)
        

        
        
        lapLoss = torch.mean(torch.abs(lap*lap))
        
        curvatures= computePrincipalCurvatures(out['HX2'], out['HY2'], out['HZ2'], grads)#[:,0]
        curvatureLosses1 = torch.relu(torch.abs(curvatures) - (rangeVals[0]/35))
        
        lapLoss = torch.mean(curvatureLosses1*curvatureLosses1)
        lapLoss2 = torch.mean(curvatures*curvatures)
        if(True):#(True):#((epoch<10) or (epoch+1)%3==0):
            loss += 1e0*lapLoss + 5e-2*lapLoss2#1e-4*lapLoss2
            
            
            
       
        
        
        # dax2 = out['a']['HX2'][:,0]
        # day2 = out['a']['HY2'][:,1]
        # daz2 = out['a']['HZ2'][:,2] 
        
        
        # a_vals = out['gradScale']['scalars']
        # a_error = 1 - a_vals
        # aLoss = torch.mean(a_error*a_error)
        # loss+= aLoss
        
        
        if(epoch>10000):#(epoch>3):#(epoch>10):
            # colLossOut = colLossFun.collision_scalar_loss(batch_input_points, grads, out['scalars'], scalarField, limVals = [x_lim, y_lim, z_lim])
            # interLoss = colLossOut['loss']
            # loss += 1e3*interLoss
            # colRecord+=interLoss
            
      
            
            colLossOut2 = colLossFun.collision_scalar_loss_far(batch_input_points, grads, out['scalars'], scalarField, limVals = [x_lim, y_lim, z_lim], n_angles=10)
            interLoss2 = colLossOut2['loss']
            if(epoch<200):
                loss += 8e3*interLoss2
            else:
                loss += 4e4*interLoss2
                
            colRecord+=interLoss2
            
            
            
            colLossOut3 = colLossFun.collision_scalar_loss_far_in(batch_input_points, grads, out['scalars'], scalarField, limVals = [x_lim, y_lim, z_lim])
            interLoss3 = colLossOut3['loss']
            if(epoch<200):
                loss += 8e3*interLoss3
            else:
                loss += 4e4*interLoss3
                
            colRecord+=interLoss3
            
            if(colLossOut is None):
                colLossOut = colLossOut2
            else:
                if(interLoss2!=0):
                    colLossOut = colLossOut2
            
            
            intError = 0

            interLoss = 0
            intError = 0
            
        else:
            interLoss = 0
            intError = 0
            
            
       # # #################  Tool Path Losses #############################
       
        if(epoch>0):
            field1grads1 = grads
            # field1scalars2 = out['scalars']
            # field1scalars2 = (2*field1scalars2.detach() - (batchMax + batchMin))*batchRangeLim/(batchMax - batchMin)
            # #field1scalars2 = torch.ones(batch_input_points.shape[0],1).to(device)
            # field1scalars2.requires_grad = True
            # newInps = torch.hstack((batch_input_points, field1scalars2))
            newInps = batch_input_points
            out = scalarField2(newInps)
            
            grads = out['grads'][:,0:3]
            
            
            grads_norm = torch.norm(grads, dim=1)
            grads_norm_grads = torch.autograd.grad(grads_norm, batch_input_points, torch.ones_like(grads_norm), create_graph=True)[0]
            # print(grads_norm_grads.shape)
            
            
            ######Project the normal of second field on tangent surface of first field#########
            field1grads1_norm = torch.norm(field1grads1, dim=1).unsqueeze(1)
            field1grads1 = field1grads1/(field1grads1_norm+ 1e-10)
            
            normal_comp = torch.sum(field1grads1*grads, dim=1).unsqueeze(1)
            project_grads = grads - normal_comp*field1grads1
            
            # field1grads1_norm = torch.norm(field1grads1, dim=1).unsqueeze(1)
            # field1grads1 = field1grads1/(field1grads1_norm + 1e-10)
            
            # normal_comp = torch.sum(grads_norm_grads*grads, dim=1).unsqueeze(1)
            # project_grads = grads_norm_grads - normal_comp*field1grads1
            
            
            normal_norm = torch.norm(grads, dim=1)
            project_norm = torch.norm(project_grads, dim=1)
            goodNorm_mask = normal_norm > 0.1
            lowNorm_mask = normal_norm <= 0.1
            targetNorm = 1*torch.ones_like(project_norm).to(device)
            gradError = project_norm - targetNorm
            
            #gradNormLoss = torch.mean(torch.abs(gradError[goodNorm_mask]*gradError[goodNorm_mask]))
            gradNormLoss = torch.mean(torch.abs(gradError*gradError))
            dx2 = out['HX2'][:,0]
            dy2 = out['HY2'][:,1]
            dz2 = out['HZ2'][:,2]
     
            curvaturesTP =  getTPcurvatureLoss(field1grads1, grads, out['HX2'], out['HY2'], out['HZ2'], 1.0)
            #print(curvatures)
            curvaturesTP = torch.relu(torch.abs(curvatures) - rangeVals[0]/(5.0))
            curvatureLoss = torch.mean(torch.abs(curvaturesTP*curvaturesTP))
            curvRecord += curvatureLoss
            # lapError = dx2 + dy2 +dz2         
            # lapLoss = torch.mean(lapError*lapError)
            if(epoch>0):
                loss += wt1*gradNormLoss + wt2*curvatureLoss
            else:
                loss += wt1*gradNormLoss
            #loss +=  1e-7*curvatureLoss 
            
            #small laplacian to remove un-necessary singularities
            # lapError = dx2 + dy2 +dz2         
            # lapLoss = torch.mean(lapError*lapError)
            
            # loss += 1e-7*lapLoss
            
            # if(torch.sum(lowNorm_mask)>0):
            #     #print("low")
            #     lapLoss2 = torch.mean(lapError[lowNorm_mask]*lapError[lowNorm_mask])
            
            #     loss += 1e-2*lapLoss2
        
            

        # # ################## Stress Points ##########################
        if(True):
            out = scalarField(batch_sPoints)
            
            sGrads = out['grads']
            sGrads = sGrads/(torch.norm(sGrads, dim=1).unsqueeze(1)+1e-10)
            sDirns = batch_sDirns
            
            dotprd = torch.sum(sGrads*sDirns, dim=1)
            stressLoss = torch.mean(dotprd*dotprd)
            
            #loss += 1e0*stressLoss
        else:
            stressLoss = 0
            
            
            
        if(epoch>0):
            field1grads2 = out['grads']
            # field1scalars2 = out['scalars']
            # field1scalars2 = (2*field1scalars2.detach() - (batchMax + batchMin))*batchRangeLim/(batchMax - batchMin)
            # #field1scalars2 = torch.ones(batch_sPoints.shape[0],1).to(device)
            # field1scalars2.requires_grad = True
            # newInps = torch.hstack((batch_sPoints, field1scalars2))
            newInps = batch_sPoints
            out1 = scalarField2(newInps)
            grads = out1['grads'][:,0:3]
        
            
            
            
            
            
            ######Project the normal of second field on tangent surface of first field#########
            field1grads2_norm = torch.norm(field1grads2, dim=1).unsqueeze(1)
            field1grads2 = field1grads2/(field1grads2_norm + 1e-10)
            
            normal_comp = torch.sum(field1grads2.detach()*grads, dim=1).unsqueeze(1)
            project_grads = grads - normal_comp*field1grads2
            
            # field1grads1_norm = torch.norm(field1grads1, dim=1).unsqueeze(1)
            # field1grads1 = field1grads1/(field1grads1_norm + 1e-10)
            
            # normal_comp = torch.sum(grads_norm_grads*grads, dim=1).unsqueeze(1)
            # project_grads = grads_norm_grads - normal_comp*field1grads1
            
            
    
            project_norm = torch.norm(project_grads, dim=1)
            targetNorm = 1*torch.ones_like(project_norm).to(device)
            gradError = project_norm - targetNorm
            
            #gradNormLoss = torch.mean(torch.abs(gradError[goodNorm_mask]*gradError[goodNorm_mask]))
            gradNormLoss = torch.mean(torch.abs(gradError*gradError))
            loss += wt1*gradNormLoss
            
            
            goodNorm_mask = project_norm > 0.1
            
            
            # curvaturesTP =  getTPcurvatureLoss(field1grads2.detach(), grads, out1['HX2'], out1['HY2'], out1['HZ2'], 1.0)
            # #print(curvatures)
            # #curvatures = torch.relu(torch.abs(curvatures) - 1/(rangeVals[0]*10.0))
            # curvatureLoss = torch.mean(torch.abs(curvaturesTP*curvaturesTP))
            # #curvRecord += curvatureLoss
        
        
            field1grads2 = field1grads2/(torch.norm(field1grads2, dim=1, keepdim=True)+1e-10)
            grads = grads/(torch.norm(grads, dim=1, keepdim=True)+1e-10)
            
            #####cross product to find tangent#########
            tangents = torch.cross(field1grads2,grads)
            tangents = tangents/(torch.norm(tangents, dim=1, keepdim=True)+1e-10)
                     
            # dotProd = normalized_tangents*batch_maxStressNormals
            # dotProd = torch.sum(dotProd, dim=1)
            # dotError = torch.abs(1-dotProd*dotProd)
            # stressLoss2 = torch.mean(dotError)
            
            cross_error = torch.cross(tangents, batch_sDirns)
            #cross_error = torch.sum(cross_error[goodNorm_mask]*cross_error[goodNorm_mask], dim=1)
            cross_error = torch.sum(cross_error*cross_error, dim=1)
            stressLoss2 = torch.mean(cross_error)
            
            # if(epoch<100):
            #     loss += 5e0*stressLoss2
            # elif(epoch<200):
            #     loss += 1e1*stressLoss2
            # else:
            #     loss += 2e1*stressLoss2
            
            
            loss += 2e1*stressLoss2
            #loss += wt2*curvatureLoss
            
            
            laps = out1['HX2'][:,0] + out1['HY2'][:,1] + out1['HZ2'][:,2]
            
            
            # dotProd = grads*field1grads2#/(gradNorm.detach()+1e-10)
            # dotProd = torch.sum(dotProd, dim=1)
            # dotError = dotProd*dotProd
            # dotError = nn.functional.relu(dotError-0.95)
            # stressLoss3 = torch.mean(dotError)
            # loss += 1e0*stressLoss3
            
            #gradNorm = torch.norm(grads, dim=1).unsqueeze(1)
            # dotProd = grads*batch_sDirns#/(gradNorm.detach()+1e-10)
            # dotProd = torch.sum(dotProd, dim=1)
            # #dotError = dotProd*dotProd
            # dotError = torch.abs(dotProd*dotProd)
            # stressLoss4 = torch.mean(dotError)
            # loss += 2e1*stressLoss4
            
        else:
            stressLoss2 = 0
            stressLoss3 = 0
            stressLoss4 = 0
            
            
            
        if(False):
            fullBoundaryOuts = scalarField(boundaryPoints)
            plLoss = platform.getPlatformPosLoss(boundaryPoints, fullBoundaryOuts['grads'], boundaryNormals)
            loss += plLoss
            # print(platform.platformDir)
            # print(platform.platformBase)
        else:
            plLoss = 0
        
        print(".",end="")
        
        loss.backward()
  
        optimizer1.step()
        optimizer1.zero_grad()
  
    
        optimizer2.step()
        optimizer2.zero_grad()
        
        optimizer3.step()
        optimizer3.zero_grad()
    
        batchLoss+=loss
        
       
      
 
    if((epoch+1)%40==0):
        if wt1<1.5e0:
            wt1*=2
            print("updating....")
        #wt2*=2
        
    
    
    
    
    
    if((epoch+1)%1==0):
        # writePath = "./parametersTest_batched_simpleBar_mid.pt"
        # checkPoint = {'model_state_dict':scalarField.state_dict()}
        # torch.save(checkPoint, writePath)
        
        # writePath2 = "./2parametersTest_batched_simpleBar_mid.pt"
        # checkPoint2 = {'model_state_dict':scalarField2.state_dict()}
        # torch.save(checkPoint2, writePath2)
        writePath = f"./parametersTest_batched_TshapeBracketNew_deep4.pt"
        checkPoint = {'model_state_dict':scalarField.state_dict()}
        torch.save(checkPoint, writePath)
        
        writePath2 = "./2parametersTest_batched_TshapeBracketNew_deep4.pt"
        checkPoint2 = {'model_state_dict':scalarField2.state_dict(), 'field1Min':batchMin, 'field1Max':batchMax}
        torch.save(checkPoint2, writePath2)
        
    if((epoch)%100==0):
        # writePath = "./parametersTest_batched_simpleBar_mid.pt"
        # checkPoint = {'model_state_dict':scalarField.state_dict()}
        # torch.save(checkPoint, writePath)
        
        # writePath2 = "./2parametersTest_batched_simpleBar_mid.pt"
        # checkPoint2 = {'model_state_dict':scalarField2.state_dict()}
        # torch.save(checkPoint2, writePath2)
        writePath = f"./Tparams/parametersTest_batched_TshapeBracketNew_deep2{epoch}.pt"
        checkPoint = {'model_state_dict':scalarField.state_dict()}
        torch.save(checkPoint, writePath)
        
        writePath2 = f"./Tparams/2parametersTest_batched_TshapeBracketNew_deep2{epoch}.pt"
        checkPoint2 = {'model_state_dict':scalarField2.state_dict(), 'field1Min':batchMin, 'field1Max':batchMax}
        torch.save(checkPoint2, writePath2)

    

    
    print("\n")
    scheduler1.step()
    scheduler2.step()
    print(lapLoss)
    print(curvRecord/iterCount)
    print(stressLoss)
    print(stressLoss2)
    print(colRecord/iterCount)
    print(batchLoss/iterCount)
    print(epoch)
    print("--")
    
    # if (True):#(epoch+1)%4 == 0:
    #     fullScalarOuts = scalarField(input_points)
    #     platfromDir = get_platformData(input_points, fullScalarOuts['scalars'], fullScalarOuts['grads'])['dir']




# writePath = "./parametersTest_batched_fertility_full2.pt"
# checkPoint = {'model_state_dict':scalarField.state_dict()}
# torch.save(checkPoint, writePath)



newPoints = meshVol.points
newPoints = torch.tensor(newPoints, dtype=torch.float32, device=device, requires_grad=True)
newPoints = (newPoints - midVals)/rangeVals
newPoints = newPoints.detach()
newPoints.requires_grad = True
newScalars = scalarField(newPoints)['scalars'].to('cpu').detach().numpy().flatten()
meshVol.point_data['scalars'] = newScalars

max_scalar = np.max(newScalars)
min_scalar = np.min(newScalars)


contourVals = np.linspace(newScalars.min()+0.0001, newScalars.max()-0.0001, 25)
contours = meshVol.contour(contourVals)

newPoints = torch.tensor(contours.points, dtype=torch.float32, device= device, requires_grad=True)
newPoints = (newPoints - midVals)/rangeVals
newScalarsTorch = scalarField(newPoints)['scalars']

# newScalarsTorch = (2*newScalarsTorch - (batchMax + batchMin))*batchRangeLim/(batchMax - batchMin)
# newScalarsTorch = newScalarsTorch.detach()
# #newScalarsTorch = torch.ones(newPoints.shape[0],1).to()
# newScalarsTorch.requires_grad = True
# newPoints = torch.hstack((newPoints, newScalarsTorch))
newPoints = newPoints.detach()
newPoints.requires_grad = True
newScalars2 = scalarField2(newPoints)['scalars'].to('cpu').detach().numpy().flatten()


contours.point_data['scalars'] = newScalars2
contourVals2 = np.linspace(newScalars2.min()+0.001, newScalars2.max()-0.001, 25)
contours2 = contours.contour(contourVals2)

# count = 0

outs = scalarField(input_points)
scalars = outs['scalars']
grads = outs['grads']
minScalar = torch.min(scalars)
maxScalar = torch.max(scalars)

thresholdScalar = minScalar + 3e-1*(maxScalar - minScalar)
baseSelectMask = scalars < thresholdScalar

pickedPoints = input_points[baseSelectMask.flatten()]*rangeVals + midVals
pickedGrads = grads[baseSelectMask.flatten()]
platformDir = torch.mean(pickedGrads, dim=0)
platformDir = platformDir/torch.norm(platformDir)
new_mesh = pv.PolyData(pickedPoints.detach().to('cpu').numpy())


dirDotProd = torch.sum(pickedPoints*(platformDir.view(1,3)), dim=1)
platformBasePointIndex = torch.nonzero(dirDotProd == torch.min(dirDotProd), as_tuple=True)[0][0].item()
platformBasePoint = pickedPoints[platformBasePointIndex]
platformBasePoint = platformBasePoint - 10*platformDir
plane = pv.Plane(center=platformBasePoint.detach().to('cpu').numpy(), direction=platformDir.detach().to('cpu').numpy(), i_size=150, j_size=150)
# platfromPos = platform.platformBase.detach()*rangeVals + midVals
# platfromDir = platform.platformDir.detach()/torch.norm(platform.platformDir.detach(), dim =1)
# plane = pv.Plane(center= platfromPos.to('cpu').numpy(), direction= platfromDir.to('cpu').numpy(), i_size=150, j_size=150)

# # contours.point_data['scalars'] = newScalars2
# # contourVals2 = np.linspace(newScalars2.min()+0.001, newScalars2.max()-0.001, 10)
# # contours2 = contours.contour(contourVals2)

# i = 858
# samplePoints = colLossOut['samples']
# samplePoints = samplePoints[i,...]
# inMask =  colLossOut['in_mask'][i,...]


# samplePoints = samplePoints.reshape(-1,3)
# inMask = inMask.flatten()


# print(samplePoints.shape)
# print(inMask.shape)


# newPointsIn = samplePoints[inMask]*rangeVals + midVals
# newPointsOut = samplePoints[~inMask]*rangeVals + midVals
# newPointsIn = newPointsIn.detach().to('cpu').numpy()
# newPointsOut = newPointsOut.detach().to('cpu').numpy()

# np.save('./arB.npy',newPointsIn)
# np.save('./arC.npy',newPointsOut)


# #pl1.add_mesh(contours)

# #


# # mask = connectionLoss(connect_boundaryPoints, connect_connectPoints, scalarField)['mask']
pl1 = pv.Plotter()
# #outs = scalarField(sPoints)['scalars']
# # grads = scalarField(boundaryPoints)['grads']
# # supportMasks = supportLoss(boundaryNormals, grads)['mask']
# # pickedPoints = boundaryPoints[supportMasks]#.reshape(-1,3)
# # newPoints = pickedPoints*rangeVals+midVals
# # new_mesh = pv.PolyData(newPoints.to('cpu').detach().numpy())
# # # #new_mesh.point_data['scalars'] = targetScalars.detach().to('cpu').numpy()
# # # newPoints2 = basePoints*rangeVals+midVals
# # # new_mesh2 = pv.PolyData(newPoints2.to('cpu').detach().numpy())


# out = scalarField(input_points)
# # supportMasks = supportLoss(boundaryNormals, grads)['mask']
pickedPoints = sPoints#[supportMasks]#.reshape(-1,3)
out1 = scalarField(pickedPoints)
newScalarsTorch = out1['scalars']

# newScalarsTorch = (2*newScalarsTorch - (batchMax + batchMin))*batchRangeLim/(batchMax - batchMin)
# newScalarsTorch = newScalarsTorch.detach()
# #newScalarsTorch = torch.ones(newPoints.shape[0],1).to()
# newScalarsTorch.requires_grad = True
# newPoints = torch.hstack((pickedPoints, newScalarsTorch))
newPoints = pickedPoints
newPoints = newPoints.detach()
newPoints.requires_grad = True
out2 = scalarField2(newPoints)

# curvs_TP = getTPcurvatureLoss(out1['grads'], out2['grads'][:,0:3], out2['HX2'], out2['HY2'], out2['HZ2'], scale = 1)
# curvs_abs = abs(curvs_TP.detach().to('cpu').numpy())
# lap = out2['HX2'][:,0] + out2['HY2'][:,1] + out2['HZ2'][:,2]
# lap = abs(lap)
# curvs_abps = lap

field1grads2 = out1['grads']/(torch.norm(out1['grads'], dim=1, keepdim=True)+1e-10)
grads = out2['grads']/(torch.norm(out2['grads'], dim=1, keepdim=True)+1e-10)

#####cross product to find tangent#########
tangents = torch.cross(field1grads2, grads)
tangents = tangents/(torch.norm(tangents, dim=1, keepdim=True)+1e-10)
         
# dotProd = normalized_tangents*batch_maxStressNormals
# dotProd = torch.sum(dotProd, dim=1)
# dotError = torch.abs(1-dotProd*dotProd)
# stressLoss2 = torch.mean(dotError)

cross_error = torch.cross(tangents, maxStressNormals)
cross_error = torch.sum(cross_error*cross_error, dim=1)
curvs_abs = cross_error.detach().to('cpu').numpy()
gradNorms = torch.norm(out2['grads'], dim=1).detach().to('cpu').numpy()


normal_comp = torch.sum(field1grads2*out2['grads'], dim=1).unsqueeze(1)
project_grads = out2['grads'] - normal_comp*field1grads2
 
 # field1grads1_norm = torch.norm(field1grads1, dim=1).unsqueeze(1)
 # field1grads1 = field1grads1/(field1grads1_norm + 1e-10)
 
 # normal_comp = torch.sum(grads_norm_grads*grads, dim=1).unsqueeze(1)
 # project_grads = grads_norm_grads - normal_comp*field1grads1
 
 

project_norm = torch.norm(project_grads, dim=1)


curvs_abs = project_norm.detach().to('cpu').numpy()
newPoints = pickedPoints*rangeVals+midVals




# # new_mesh = pv.PolyData(newPoints.to('cpu').detach().numpy())
# # # #new_mesh.point_data['scalars'] = targetScalars.detach().to('cpu').numpy()
# # # newPoints2 = basePoints*rangeVals+midVals
# # # new_mesh2 = pv.PolyData(newPoints2.to('cpu').detach().numpy())

# intError = intersectionLoss2(input_points, out['grads'], out['scalars'], scalarField, 0.0, x_lim, y_lim, z_lim)
# mask = intError['mask']
#pickedPoints = input_points#[mask]#.reshape(-1,3)
# pickedPoints = boundaryPoints
#newPoints = pickedPoints*rangeVals+midVals
new_mesh = pv.PolyData(newPoints.to('cpu').detach().numpy())
new_mesh.point_data['scalars'] = curvs_abs
# # #new_mesh.point_data['scalars'] = targetScalars.detach().to('cpu').numpy()
# # newPoints2 = basePoints*rangeVals+midVals
# # new_mesh2 = pv.PolyData(newPoints2.to('cpu').detach().numpy())
# # #pl1.add_mesh(new_mesh2, color='green')
#pl1.add_mesh(new_mesh,color='red')
#pl1.add_mesh(plane)
#pl1.add_mesh(new_mesh)
pl1.add_mesh(contours, cmap='rainbow')
pl1.add_mesh(contours2, line_width=10, color='black')
# pl1.add_mesh(mesh, style='wireframe')








        




