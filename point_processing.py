from utils.path_utils import distance_to_origin


def process_2d_pcd(x, y, z, doppler):
    """
    process 2d point cloud data
    :param x:
    :param y:
    :param doppler:
    :return:
    """

    if len(x) == 0 or len(y) == 0 or len(doppler) == 0:
        raise Exception("point_processing: process_2d_pcd: no point to process")

    point_list = []
    point_distance_list = []
    x_disp = 0
    y_disp = 0

    # generate point list (x, y, speed)
    for i in range(len(x)):
        point_list.append((x[i], y[i], doppler[i]))
        point_distance_list.append(distance_to_origin(x[i], y[i]))

    # sort by the absolute speed in range
    point_list.sort(key=lambda x: abs(x[2]), reverse=True)

    maxspd_point = point_list[0]

    y_disp = maxspd_point[2]

    # get the closest point
    sorted_point_distance_list = point_distance_list.copy()
    sorted_point_distance_list.sort(key=lambda x: x, reverse=True)

    closest_point = point_list[point_distance_list.index(sorted_point_distance_list[0])]

    # remove outliers
    # dist_outliers = get_outliers(point_distance_list)
    # point_outliers = []
    #
    # for dol in dist_outliers:
    #     point_outliers.append(point_list[point_distance_list.index(dol)])
    #
    # for pol in point_outliers:
    #
    #     point_list.remove(pol)
    #
    # x_ol_removed = []
    # y_ol_removed = []
    # # z_ol_removed = []
    #
    # # create x, y list with outliers removed
    # for point in point_list:
    #     x_ol_removed.append(point[0])
    #     y_ol_removed.append(point[1])

    # resolve x displacement

    # user linear regression
    # fit = np.polyfit(x, y, 1)
    #
    # slope, intercept = fit[1], fit[0]
    #
    # fit_x_line = np.arange(10.0).tolist()  # get list [0, 0.1, 0.2, 0.3,... 1.0]
    # fit_y_line = []
    #
    # for fit_x in fit_x_line:
    #     fit_y_line.append(fit_x * slope + intercept)

    return closest_point# x_disp, y_disp, fit_x_line, fit_y_line #, x_ol_removed, y_ol_removed
