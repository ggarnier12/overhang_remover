#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 15:37:45 2020

@author: gildas garnier
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
#import math
#from stl import mesh
import pymesh
import stl_render

bottom_tol=0.1

###debug functions to view in 2D
import matplotlib.pyplot as plt

def yzVerticesPlot(initial_vertices,projected_vertices):
    plt.figure()
    Vertice2DPlot(projected_vertices,'ro',2,3)
    Vertice2DPlot(initial_vertices,'bo',2,3)
    plt.show()

def xyVerticesPlot(initial_vertices,projected_vertices):
    plt.figure()
    Vertice2DPlot(projected_vertices,'ro',1,2)
    Vertice2DPlot(initial_vertices,'bo',1,2)
    plt.show()
    
def Vertice2DPlot(vertices,style,dim1,dim2):
    vertices=np.array(vertices)
    y_vertices=[c[dim1-1] for c in vertices]
    z_vertices=[c[dim2-1] for c in vertices]
    plt.plot(y_vertices,z_vertices, style)
#############################
    

def z0Cut(input_mesh):
    """this function sets z vectrices negative or lower than tolerance to zero"""
    vertices=input_mesh.vertices.copy()
    for ive, ve in enumerate(vertices):
        if ve[2]<bottom_tol:
            ve[2]=0
    output_mesh=pymesh.form_mesh(vertices, input_mesh.faces)
    return output_mesh

def bottomTriangles(allFaces, allVertices):
    """this functions returns the list of triangles of the mesh at z=0"""
    listOfFaces=[]
    
    for itri,face in enumerate(allFaces):
        if max(allVertices[face[0]][2],
               allVertices[face[1]][2],
               allVertices[face[2]][2])<bottom_tol:
            listOfFaces.append(itri)
#    for itri,tri in enumerate(iniMesh.vectors):
#        if max(tri[0][2],tri[1][2],tri[2][2])<bottom_tol:
#            listOfTriangles.append(itri)
    return listOfFaces

def removeElementsFromList(l1,l2):
    """this function returns a list which is the first input list minus the elements from the second list""" 
    l2s = set(l2)
    l3 = [x for x in l1 if x not in l2s]
    return l3

def vectAngle(v1: np.ndarray,
          v2: np.ndarray):
    """this function return the angle between two vectors in degrees using arcos, from 0 to pi"""
    return np.arccos(np.dot(v1/np.linalg.norm(v1), v2/np.linalg.norm(v2)))

def overhanging_from_lists(allFaces,allVertices,allNormals,max_angle):
    """this function list the triangles indexes from the mesh that are overhanging based on max_angle in rad"""
    listOfFacesIndex=[]
    for iv,v in enumerate(allNormals):
        vAngle=vectAngle(v, np.array([0,0,-1]))
        if vAngle<np.pi/2-max_angle:
            listOfFacesIndex.append(iv)
    listOfFacesIndex=removeElementsFromList(listOfFacesIndex,
                                       bottomTriangles(allFaces,allVertices))
    return listOfFacesIndex

def overhanging_from_mesh(mesh,max_angle):
    mesh.add_attribute("face_normal")
    normals = mesh.get_attribute("face_normal")
    normals=[[normals[_*3],normals[_*3+1],normals[_*3+2]] for _ in range(int(len(normals)/3))]

    return overhanging_from_lists(mesh.faces,
                                  mesh.vertices,
                                  normals,
                                  max_angle)

def projectVerticeOnCone(vertice: np.ndarray,
                         cone: dict):
    """this fonction return the vertice projected on the cone. cone Angle = angle from direction to surface in rad"""
    #define the normal to the plan of projection (including coneApex, coneDirection and vertice to project)
    normalToProjectionPlan=np.cross(vertice-cone['apex'],cone['direction'])
    normalToProjectionPlan=normalToProjectionPlan/np.linalg.norm(normalToProjectionPlan)
    #the out of plan part of vertice coordinates is extracted, it will stay unchanged
#    oopVertice=np.dot(vertice,normalToProjectionPlan)*normalToProjectionPlan
    #calculate in plane part of resulting projection
    #first define both edges of the cone within the plane
    rotation1 = R.from_rotvec(cone['angle'] * normalToProjectionPlan)
    rotation2 = R.from_rotvec(-cone['angle'] * normalToProjectionPlan)
    ConeEdge1=rotation1.apply(cone['direction']) 
    ConeEdge2=rotation2.apply(cone['direction']) 
    #calculate the distance from each edge to find closest for projection
    distanceToEdge1=np.linalg.norm(np.cross(ConeEdge1,vertice-cone['apex']))
    distanceToEdge2=np.linalg.norm(np.cross(ConeEdge2,vertice-cone['apex']))
    #calculate in plane part of projected vertice
    if distanceToEdge1<distanceToEdge2:
        ipVertice=cone['apex']+np.dot(vertice-cone['apex'],ConeEdge1)*ConeEdge1
    else:
        ipVertice=cone['apex']+np.dot(vertice-cone['apex'],ConeEdge2)*ConeEdge2

    #TODO if the projection goes below z=0, the position should be ajusted to keep the same angle but at z=0            
    return ipVertice#oopVertice+ipVertice

def FollowEdgePath(mesh,
                   startVerticeIndex: int,
                   direction: np.ndarray,
                   angleCriteria: float):
    """this function follows the edges closest to the direction from the startVertice
    until they reach a vertice forming an angle from startVertice with direction, 
    lower than the angle criteria"""
    mesh.enable_connectivity()
    steps=[]
    steps.append(startVerticeIndex)
    angle=np.pi
    while angle>angleCriteria:
        currentStep=steps[len(steps)-1]
        potentialsNextStep=mesh.get_vertex_adjacent_vertices(currentStep)
        directions=[vectAngle(direction,mesh.vertices[vi]-mesh.vertices[currentStep]) 
                        for vi in potentialsNextStep]
        if min(directions)>=np.pi/2:
            #TODO in this case we should place a specific vertice at the very bottom to support the structure
            #for now, it is just a projection like any other
            break
        nextStep=potentialsNextStep[np.argmin(directions)]
        steps.append(nextStep)
        angle=vectAngle(direction,mesh.vertices[nextStep]-mesh.vertices[startVerticeIndex])
    return steps

def projectVerticesAlongPath(mesh,
                             startVerticeIndex: int,
                             direction: np.ndarray,
                             angleCriteria: float,
                             projectionsList: dict):
    """this function projects all vertices along the path closest to the direction
    starting from the startVertice. It does not project vertices if they were
    previously projected. it appends projected vertices to the projectionList 
    which is a dictionnay with original vertice index as key and projected 
    vertice coordinates as values"""
    
    steps=FollowEdgePath(mesh,startVerticeIndex,direction,angleCriteria)
    cone={
        'apex':mesh.vertices[startVerticeIndex],
        'angle':angleCriteria,
        'direction':direction}
    projectionsList[steps[0]]=mesh.vertices[steps[0]]
    for step in steps[1:-1]:
        if not step in projectionsList.keys():
            projectionsList[step]=projectVerticeOnCone(mesh.vertices[step],cone)
    projectionsList[steps[-1]]=mesh.vertices[steps[-1]]

    return projectionsList

def buildUnderOverhangingVertices(mesh,
                                  direction: np.ndarray,
                                  angleCriteria: float):
    """this function gets all vertices from overhanging faces.
    Starting from the top one, it creates projections of the vertices below 
    along the steepiest path to support it. Then it does the same for the 
    remaining highest vertice and so on until there are no more overhanging
    vertice that has not been projected"""
    #get the list of overhanging faces
    overhangingFacesIndex=overhanging_from_mesh(mesh,angleCriteria)
    #get the list of vertices from overhanging faces
    overhangingVerticesIndex=[]
    for iface in overhangingFacesIndex:
        for ivertice in mesh.faces[iface]:
            if not ivertice in overhangingVerticesIndex:
                overhangingVerticesIndex.append(ivertice)
    #sort this list in the direction, building a dictionnary with vertice index as key and position as value
    heightOfVertices={}
    for ivertice in overhangingVerticesIndex:
        heightOfVertices[ivertice]=np.dot(mesh.vertices[ivertice],direction)
    sortedVertices={k: v for k, v in sorted(heightOfVertices.items(), key=lambda item: item[1])}
    #from the first to the last, if not already projected, project along the path below the vertice
    projectionsList={}
    for ivertice in sortedVertices:
        if not ivertice in projectionsList.keys():
            projectVerticesAlongPath(mesh, 
                                     ivertice, 
                                     direction, angleCriteria,
                                     projectionsList)
#            xyVerticesPlot([mesh.vertices[_] for _ in projectionsList.keys()],
#                           [_ for _ in projectionsList.values()])
#            yzVerticesPlot([mesh.vertices[_] for _ in projectionsList.keys()],
#                           [_ for _ in projectionsList.values()])
            #print(projectionsList)
    #now we have all overhanging vertices with projections 
    #(plus the one below needed to support them) 
    #Next step is to build the mesh using these
    allVertices=mesh.vertices.copy()
    projectedVertices=np.array([v for v in projectionsList.values()])
    if len(projectedVertices)>0:
        allVertices=np.concatenate([allVertices,projectedVertices])
    #dictionnary of initial vertice index and projected vertice index in allVertices
    projectionsIndex={t[0]: i+len(mesh.vertices) 
                    for i, t in enumerate(projectionsList.items())}
    #if all vertices have been projected, add a face with all the projected vertices but the original face is not copied
    #if no vertice from an initial face has been projected, copy the face in the new mesh
    #if some vertice were projected but not all, copy the original face and keep the list of projected vertices as a hollow quad
    hollowQuadOriginalVertices=[]
    newMeshFaces=[]
    for face in mesh.faces:
        verticesProjected=[]
        nVerticesProjected=0
        for ivertice in face:
            if ivertice in projectionsList.keys():
                nVerticesProjected+=1
                verticesProjected.append(ivertice)
        if nVerticesProjected==0:
            newMeshFaces.append(face)
        elif nVerticesProjected==3:
            newMeshFaces.append([
                    projectionsIndex[face[0]],
                    projectionsIndex[face[1]],
                    projectionsIndex[face[2]]
                    ])
        else:
            newMeshFaces.append(face)
            edges=[[face[0],face[1]],
                   [face[1],face[2]],
                   [face[2],face[0]]]
            for edge in edges:
                #TODO what does it mean if only one vertice of an edge was projected ?
                # so far in this case we do nothing
                if (edge[0] in verticesProjected) and (edge[1] in verticesProjected):
                    hollowQuadOriginalVertices.append(edge)
    #for each hollow quad, split it in two triangles and add to mesh
    #TODO check if resulting faces are overhaning and if yes, rework it !
    for edge in hollowQuadOriginalVertices:
        newMeshFaces.append([edge[1],edge[0],projectionsIndex[edge[1]]])
        newMeshFaces.append([projectionsIndex[edge[0]],projectionsIndex[edge[1]],edge[0]])
    #build the resulting mesh from vertices (initial + projected) and new list of faces
    newMesh=pymesh.form_mesh(allVertices, np.array(newMeshFaces))
    return newMesh
    
    
###Former strategy based on faces projection
#def projectFacesOnCone(facesToProject,
#                       allFaces: np.ndarray,
#                       allVertices: np.ndarray,
#                       cone: dict):
#    """this function projects the faces from allFaces with indices listed in facesToProject. 
#    It creates the new vertices for these projected faces.
#    It also lists the resulting quad hollow faces"""
#    #TODO deal with free edges and hollow quads
#    #TODO list vertices projection so as not to duplicate and reuse
#    hollowquads=[]
#    for face in facesToProject:
#        for _ in range(3):
#            allVertices.append(
#                    projectVerticeOnCone(allVertices[allFaces[face][_]], cone))
##            edge=[allFaces[face][_],allFaces[face][(_+1)%3]]
##            if not edge in freeEdges:
##                hollowquads.append([edge,[]])
#        lav=np.shape(allVertices) [0]
#        
#        hollowquads.append([])
#        allFaces[face]=[lav-3,lav-2,lav-1]
#        
#    
#    return allFaces,allVertices#,hollowquads



#z0Mesh=z0Cut(initialMesh)

#z0Mesh.add_attribute("face_normal")
#normals = z0Mesh.get_attribute("face_normal")
#normals=[[normals[_*3],normals[_*3+1],normals[_*3+2]] for _ in range(int(len(normals)/3))]
#
#print(overhanging(initialMesh.faces,initialMesh.vertices,normals,70/180*np.pi))

#for iT in overhanging(your_mesh,70):
#    print(your_mesh.vectors[iT])
#angleCriteria=45/180*np.pi#np.pi/4
#
##projection test
#cone={}
#cone['apex']=np.array([10,0,20])
#cone['angle']=np.pi*0.5-angleCriteria
#cone['direction']=np.array([0,0,-1])
#vertice=np.array([10,4,19])
##coneApex=np.array([0,0,1])
##coneAngle=np.pi/4
##coneDirection=np.array([0,0,-1])
#print(projectVerticeOnCone(vertice,cone))
#startVertice=50
#direction=cone['direction']
#mesh=z0Mesh
#projectionsList={}
#steps=FollowEdgePath(mesh,startVertice,direction,np.pi/4.)
#projectVerticesAlongPath(mesh,startVertice,direction,np.pi/4.,projectionsList)
#
#newMesh=buildUnderOverhangingVertices(mesh,direction,angleCriteria)
#
#pymesh.save_mesh(mesh=newMesh,filename='out.stl')

def test_case(testname):
    initialMesh = pymesh.load_mesh(testname+'.stl')
#    initialMesh.enable_connectivity()
    initialMesh=z0Cut(initialMesh)
    angleCriteria=45/180*np.pi#np.pi/4
    direction=np.array([0,0,-1])
    newMesh=buildUnderOverhangingVertices(initialMesh,direction,angleCriteria)
    out_filename=testname+'-out.stl'
    pymesh.save_mesh(mesh=newMesh,filename=out_filename)
    stl_render.stl_render(out_filename)

    
def test_cases():
    test_case('tests/arch')
    test_case('tests/cube_mesh')
    test_case('tests/flat_ring')  
    test_case('tests/lateral_cone')
    test_case('tests/Mask_breather_V4')
    

