import numpy as np

def ransac_plane_fit(points, threshold=0.01, num_ransac_points = 3, num_iterations=1000, min_inliers=100):
    best_plane = None
    best_inliers = []

    num_points = points.shape[0]

    for _ in range(num_iterations):
        indices = np.random.choice(num_points, num_ransac_points, replace=False)
        sample = points[indices]

        p1, p2, p3 = sample
        normal = np.cross(p2 - p1, p3 - p1)
        if np.linalg.norm(normal) == 0:
            continue

        a, b, c = normal
        d = -np.dot(normal, p1)

        numerators = np.abs((points @ normal) + d)
        denominator = np.linalg.norm(normal)
        distances = numerators / denominator

        inlier_indices = np.where(distances < threshold)[0]

        if len(inlier_indices) > len(best_inliers):
            best_inliers = inlier_indices
            best_plane = (a, b, c, d)

    if len(best_inliers) < min_inliers:
        return None, None

    inlier_points = points[best_inliers]
    return best_plane, inlier_points, best_inliers





def segment_planes_ransac(points,
                          distance_threshold=0.01,
                          num_iterations=1000,
                          min_ratio=0.01,
                          min_num_points=100,
                          max_planes=10):

    planes = []
    rest_points = points.copy()
    total_points = points.shape[0]

    while len(rest_points) > min_num_points and len(planes) < max_planes:
        result = ransac_plane_fit(rest_points,
                                  threshold=distance_threshold,
                                  num_iterations=num_iterations,
                                  min_inliers=max(min_ratio * total_points, min_num_points))

        if result is None:
            break

        plane_model, inlier_points, inlier_indices = result

        planes.append((plane_model, inlier_points))

        mask = np.ones(len(rest_points), dtype=bool)
        mask[inlier_indices] = False
        rest_points = rest_points[mask]

    return planes, rest_points
