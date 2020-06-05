import math


def create_geographic_chunks(longitude=None, latitude=None, geographic_chunk_size=0.5):
    """Spatially chunk a parameter set defined by latitude and longitude.

    Parameters
    ----------
    longitude: list-like
        Longitude range to split
    latitude: list-like
        Latitude range to split

    Returns
    -------
    geographic_chunks: list of dicts
        A list of dicts mapping longitude and latitude to 2-tuples of their ranges
        for each chunk.
    """

    assert latitude and longitude, "Longitude and latitude are both required kwargs."
    assert len(latitude) == 2 and latitude[1] >= latitude[0], \
        "Latitude must be a tuple of length 2 with the second element greater than or equal to the first."
    assert len(longitude) == 2 and longitude[1] >= longitude[0], \
        "Longitude must be a tuple of length 2 with the second element greater than or equal to the first."

    square_area = (latitude[1] - latitude[0]) * (longitude[1] - longitude[0])
    geographic_chunks = max(1, math.ceil(square_area / geographic_chunk_size))

    #we're splitting accross latitudes and not longitudes
    #this can be a fp value, no issue there.
    latitude_chunk_size = (latitude[1] - latitude[0]) / geographic_chunks
    latitude_ranges = [(latitude[0] + latitude_chunk_size * chunk_number,
                        latitude[0] + latitude_chunk_size * (chunk_number + 1))
                       for chunk_number in range(geographic_chunks)]
    longitude_ranges = [longitude for __ in latitude_ranges]

    return [{'longitude': pair[0], 'latitude': pair[1]} for pair in zip(longitude_ranges, latitude_ranges)]

