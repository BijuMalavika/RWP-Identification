import vtk
import numpy as np
from sklearn import cluster
from scipy import spatial
import math
import xarray as xr
import pyvista as pv


def addMaxData(DS):

    da = DS.v[0]
    scalar_values = da.values

    r, c = scalar_values.shape
    check = np.zeros((r, c))
    is_max = np.zeros((r, c))
    vertex_identifiers = np.zeros(r*c)

    lon1 = DS.longitude
    lat1 = DS.latitude
    lon = lon1.values
    lat = lat1.values
    rect = pv.RectilinearGrid(lon, lat)
    scalar = scalar_values.ravel()
    rect.point_arrays['v'] = scalar

    count = 0
    k = 0

    for i in range(r):
        for j in range(c):

            vertex_identifiers[k] = k+1
            k += 1
            max_flag = 1

            if(check[i][j] == 1):
                continue

            else:
                if(j == 0):
                    for x in [i-1, i, i+1]:
                        for y in [c-1, j, j+1]:
                            if((0 <= x < r) and (0 <= y < c)):
                                if(scalar_values[x][y] < scalar_values[i][j]):
                                    max_flag = 0
                                else:
                                    check[x][y] = 1

                if(j == c-1):
                    for x in [i-1, i, i+1]:
                        for y in [j-1, j, 0]:
                            if((0 <= x < r) and (0 <= y < c)):
                                if(scalar_values[x][y] < scalar_values[i][j]):
                                    max_flag = 0
                                else:
                                    check[x][y] = 1

                else:
                    for x in [i-1, i, i+1]:
                        for y in [j-1, j, j+1]:
                            if((0 <= x < r) and (0 <= y < c)):
                                if(scalar_values[x][y] < scalar_values[i][j]):
                                    max_flag = 0
                                else:
                                    check[x][y] = 1

            if(max_flag == 1):
                is_max[i][j] = 1
                check[i][j] = 1

    cell_number = rect.GetNumberOfCells()
    cell_id = np.zeros(cell_number)
    for i in range(cell_number):
        cell_id[i] = i

    rect.point_arrays['is max'] = is_max.ravel()
    rect.point_arrays['Vertex_id'] = vertex_identifiers
    rect.cell_arrays["Cell_V"] = cell_id
    return(rect)



def extractPosMaxIds(scalar_field,value):
    pos_max_ids = vtk.vtkIdTypeArray()
    num_pts = scalar_field.GetNumberOfPoints()
    is_max_arr = scalar_field.GetPointData().GetArray("is max")
    scalar_arr = scalar_field.GetPointData().GetArray("v")
    count = 0
    for i in range(num_pts):
        if(is_max_arr.GetTuple1(i) == 1 and scalar_arr.GetTuple1(i) <= value):
            pos_max_ids.InsertNextValue(i)
        
    print(count)
    return pos_max_ids


def extractSelectionIds(scalar_field, id_list):
    selectionNode = vtk.vtkSelectionNode()
    selectionNode.SetFieldType(1)
    selectionNode.SetContentType(4)
    selectionNode.SetSelectionList(id_list)
    selection = vtk.vtkSelection()
    selection.AddNode(selectionNode)
    extractSelection = vtk.vtkExtractSelection()
    extractSelection.SetInputData(0, scalar_field)
    extractSelection.SetInputData(1, selection)
    extractSelection.Update()
    return extractSelection.GetOutput()


def clusterMax(scalar_field, connectivity_clipped_scalar_field, max_points):
    # import scalar field and critical point data objects
    scalar_field = connectivity_clipped_scalar_field
    maxima_points = max_points
    base_field = scalar_field

    geometryFilter = vtk.vtkGeometryFilter()
    geometryFilter.SetInputData(scalar_field)
    geometryFilter.Update()
    scalar_field = geometryFilter.GetOutput()

    #print(scalar_field)

    triangleFilter = vtk.vtkTriangleFilter()
    triangleFilter.SetInputData(scalar_field)
    triangleFilter.Update()
    scalar_field = triangleFilter.GetOutput()

    #print(scalar_field)

    maxima_point_id = maxima_points.GetPointData().GetArray("vtkOriginalPointIds")
    num_points = maxima_points.GetNumberOfPoints()
    # maxima_point_id=maxima_points.GetPointData().GetArray("VertexIdentifiers")
    # print(maxima_point_id)

    maxima_regions = maxima_points.GetPointData().GetArray("RegionId")

    point_region_id = scalar_field.GetPointData().GetArray("RegionId")
    num_regions = int(np.max(point_region_id)+1)

    dist_matrix = np.full((num_points, num_points), 400)

    dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
    dijkstra.SetInputData(scalar_field)

    #region_distance_array=[[[0 for col in range(0)]for row in range(0)]for clusters in range(num_regions)]

    locator = vtk.vtkCellLocator()
    locator.SetDataSet(base_field)
    locator.BuildLocator()
    cellIds = vtk.vtkIdList()

    cell_v = base_field.GetCellData().GetArray("Cell_V")

    co_ords = np.empty((0, 3))
    for i in range(num_points):
        co_ords = np.append(co_ords, [maxima_points.GetPoint(i)], axis=0)

    for i in range(num_points):
        for j in range(i+1, num_points):
            min_v = 1000
            max_v = -1000
            av_v = 0
            p0 = [0, 0, 0]
            p1 = [0, 0, 0]
            dist = 0.0
            region_1 = maxima_regions.GetTuple1(i)
            region_2 = maxima_regions.GetTuple1(j)
            if(region_1 != region_2):
                continue
            dijkstra.SetStartVertex(int(maxima_point_id.GetTuple1(i)))
            dijkstra.SetEndVertex(int(maxima_point_id.GetTuple1(j)))
            dijkstra.Update()
            pts = dijkstra.GetOutput().GetPoints()
            for ptId in range(pts.GetNumberOfPoints()-1):
                pts.GetPoint(ptId, p0)
                pts.GetPoint(ptId+1, p1)
                dist += math.sqrt(vtk.vtkMath.Distance2BetweenPoints(p0, p1))
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist
            locator.FindCellsAlongLine(co_ords[i], co_ords[j], 0.001, cellIds)
            for k in range(cellIds.GetNumberOfIds()):
                if(cell_v.GetTuple1(cellIds.GetId(k)) > max_v):
                    max_v = cell_v.GetTuple1(cellIds.GetId(k))
                    max_cell_id = cellIds.GetId(k)
            dist_matrix[i][j] = dist_matrix[i][j]+max_v
            dist_matrix[j][i] = dist_matrix[i][j]
            # print(dist)

    region_array = [[0 for col in range(0)]for row in range(num_regions)]
    cluster_assign = np.full(num_points, 0)

    median_dist = -np.median(dist_matrix)

    print('dist matrix computed')

    for i in range(num_points):
        # print(point_region_id.GetTuple1(int(maxima_point_id.GetTuple1(i))))
        region_array[int(point_region_id.GetTuple1(
            int(maxima_point_id.GetTuple1(i))))].append(i)

    prev_max = 0

    for k in range(num_regions):
        if(len(region_array[k]) == 1):
            cluster_assign[region_array[k][0]] = prev_max
            prev_max += 1
            continue
        if(len(region_array[k]) == 2):
            cluster_assign[region_array[k][0]] = prev_max
            cluster_assign[region_array[k][1]] = prev_max
            prev_max += 1
            continue

    #    print(len(region_array[k]))
        num_cluster = int(len(region_array[k]))
        new_dist = np.full((num_cluster, num_cluster), 0)
        # print(new_dist)

        for i in range(num_cluster):
            for j in range(i+1, num_cluster):
                new_dist[i][j] = dist_matrix[region_array[k]
                                             [i]][region_array[k][j]]
                new_dist[j][i] = new_dist[i][j]

    # print(new_dist)

        if(num_cluster == 0):
            continue

        sim_matrix = np.negative(new_dist)

        # print(sim_matrix)

        af_clustering = cluster.AffinityPropagation(preference=np.full(
            num_cluster, median_dist/5.0), affinity='precomputed')
        af_clustering.fit(sim_matrix)
        clusters = af_clustering.labels_ + prev_max
        prev_max = np.max(clusters)+1
        # print(clusters)

        for i in range(num_cluster):
            cluster_assign[region_array[k][i]] = clusters[i]
        # print(cluster_assign)

    cluster_id = vtk.vtkIntArray()
    cluster_id.SetNumberOfComponents(1)
    cluster_id.SetNumberOfTuples(num_points)
    cluster_id.SetName("Cluster ID")

    # print(cluster_assign)

    for i in range(num_points):
        cluster_id.SetTuple1(i, cluster_assign[i])

    # clustered_output=self.GetOutput()
    maxima_points.GetPointData().AddArray(cluster_id)
    # clustered_output.ShallowCopy(maxima_points)
    return maxima_points
    # print(dijkstra.GetOutput())


dataDIR = 'Downloads/souders_v_1.nc'
DS = xr.open_dataset(dataDIR)
scalar_field = addMaxData(DS)

#print(scalar_field)

scalar_field = scalar_field.point_data_to_cell_data(pass_point_data=True)

#print(scalar_field)

clipped_scalar_field = scalar_field.clip_scalar(scalars='v', value=0, invert=True)

#print(clipped_scalar_field)

connectivity_clipped_scalar_field = clipped_scalar_field.connectivity()

#print(connectivity_clipped_scalar_field)

max_points = extractSelectionIds(connectivity_clipped_scalar_field, extractPosMaxIds(connectivity_clipped_scalar_field,-5))

#print(max_points)

max_points = clusterMax(scalar_field, connectivity_clipped_scalar_field, max_points)

print(max_points)

vtuFileWriter=vtk.vtkXMLUnstructuredGridWriter()
vtuFileWriter.SetInputDataObject(max_points)
vtuFileWriter.SetFileName('clustered_min1.vtu')
vtuFileWriter.Update()

