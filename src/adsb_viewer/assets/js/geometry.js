export function heightFor(sample) {
  return Number(sample.z_m_visual ?? sample.z_m ?? 0);
}

export function heightForPoint(point) {
  return Number(point.z_m_visual ?? point.z_m ?? 0);
}

export function cartesianFromLLH(lon, lat, h) {
  return Cesium.Cartesian3.fromDegrees(lon, lat, h);
}

export function cartesianFromSample(sample) {
  return cartesianFromLLH(sample.lon, sample.lat, heightFor(sample));
}

export function cartesianFromPoint(point, zOffsetM = 0) {
  return cartesianFromLLH(
    point.lon,
    point.lat,
    heightForPoint(point) + zOffsetM,
  );
}

export function offsetLngLatMeters(lon, lat, eastM, northM) {
  const earthRadiusM = 6371000.0;
  const latRad = Cesium.Math.toRadians(lat);

  const dLat = northM / earthRadiusM;
  const dLon = eastM / (earthRadiusM * Math.max(1e-9, Math.cos(latRad)));

  return {
    lon: lon + Cesium.Math.toDegrees(dLon),
    lat: lat + Cesium.Math.toDegrees(dLat),
  };
}

export function deltaEastNorthMeters(a, b) {
  const ax = Number(a.x_m);
  const ay = Number(a.y_m);
  const bx = Number(b.x_m);
  const by = Number(b.y_m);

  if (
    Number.isFinite(ax) && Number.isFinite(ay) &&
    Number.isFinite(bx) && Number.isFinite(by) &&
    Math.hypot(bx - ax, by - ay) > 1e-6
  ) {
    return {
      east: bx - ax,
      north: by - ay,
    };
  }

  const earthRadiusM = 6371000.0;
  const meanLatRad = Cesium.Math.toRadians((a.lat + b.lat) * 0.5);

  return {
    east: Cesium.Math.toRadians(b.lon - a.lon) * earthRadiusM * Math.cos(meanLatRad),
    north: Cesium.Math.toRadians(b.lat - a.lat) * earthRadiusM,
  };
}

export function shiftedPointAlongSegment(point, eastM, northM, zM) {
  const ll = offsetLngLatMeters(point.lon, point.lat, eastM, northM);
  return Cesium.Cartesian3.fromDegrees(ll.lon, ll.lat, zM);
}

export function vectorGlyphPositionsFromComponents(
  sample,
  eastComponent,
  northComponent,
  scaleMPerUnit,
  minLengthM,
  maxLengthM,
  zOffsetM = 2.0,
  originEastM = 0.0,
  originNorthM = 0.0,
  upComponent = 0.0,
  originUpM = 0.0,
) {
  const origin = cartesianFromSample(sample);
  const degenerate = {
    valid: false,
    segment: [origin, origin],
  };

  const east = Number(eastComponent || 0);
  const north = Number(northComponent || 0);
  const up = Number(upComponent || 0);

  if (!Number.isFinite(east) || !Number.isFinite(north) || !Number.isFinite(up)) {
    return degenerate;
  }

  const mag = Math.hypot(east, north, up);
  if (!Number.isFinite(mag) || mag < 1e-6) {
    return degenerate;
  }

  const unitEast = east / mag;
  const unitNorth = north / mag;
  const unitUp = up / mag;
  const lengthM = Math.max(
    Number(minLengthM || 0),
    Math.min(Number(maxLengthM || 1e9), mag * Number(scaleMPerUnit || 1)),
  );

  const z = heightFor(sample) + Number(zOffsetM || 0);

  const baseOffsetEast = Number(originEastM || 0);
  const baseOffsetNorth = Number(originNorthM || 0);
  const baseOffsetUp = Number(originUpM || 0);
  const startLL = offsetLngLatMeters(sample.lon, sample.lat, baseOffsetEast, baseOffsetNorth);
  const tipLL = offsetLngLatMeters(
    sample.lon,
    sample.lat,
    baseOffsetEast + unitEast * lengthM,
    baseOffsetNorth + unitNorth * lengthM,
  );

  const start = Cesium.Cartesian3.fromDegrees(startLL.lon, startLL.lat, z + baseOffsetUp);
  const tip = Cesium.Cartesian3.fromDegrees(tipLL.lon, tipLL.lat, z + baseOffsetUp + unitUp * lengthM);

  return {
    valid: true,
    segment: [start, tip],
  };
}
