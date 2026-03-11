/**
 * Entity state serialization for the Python brain system.
 * Observes and serializes entity states from mineflayer bot.
 */

const HOSTILE_TYPES = new Set([
  'zombie', 'skeleton', 'creeper', 'spider',
  'enderman', 'witch', 'blaze', 'ghast',
  'slime', 'phantom', 'drowned', 'husk',
  'stray', 'cave_spider', 'wither_skeleton',
]);

function isHostile(entityType) {
  const normalized = (entityType || '').toLowerCase().replace(/ /g, '_');
  return HOSTILE_TYPES.has(normalized);
}

function serializeEntity(entity, refPos) {
  const pos = entity.position;
  const dx = pos.x - refPos.x;
  const dy = pos.y - refPos.y;
  const dz = pos.z - refPos.z;
  const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

  return {
    id: entity.id,
    type: entity.name || entity.type || 'unknown',
    x: pos.x,
    y: pos.y,
    z: pos.z,
    health: entity.health || 20,
    isHostile: isHostile(entity.name || entity.type),
    distance,
  };
}

function getVisibleEntities(bot, maxDistance = 48) {
  const refPos = bot.entity.position;
  const entities = [];

  for (const entity of Object.values(bot.entities)) {
    if (entity === bot.entity) continue;
    if (!entity.position) continue;

    const dx = entity.position.x - refPos.x;
    const dy = entity.position.y - refPos.y;
    const dz = entity.position.z - refPos.z;
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

    if (dist <= maxDistance) {
      entities.push(serializeEntity(entity, refPos));
    }
  }

  // Sort by distance
  entities.sort((a, b) => a.distance - b.distance);

  return entities;
}

function getPlayerState(bot) {
  const pos = bot.entity.position;
  const vel = bot.entity.velocity;

  return {
    health: bot.health,
    food: bot.food,
    position: { x: pos.x, y: pos.y, z: pos.z },
    velocity: { x: vel.x, y: vel.y, z: vel.z },
    yaw: bot.entity.yaw,
    pitch: bot.entity.pitch,
    timeOfDay: bot.time.timeOfDay,
    lightLevel: bot.blockAt(pos) ? bot.blockAt(pos).light : 15,
    onGround: bot.entity.onGround,
    isInWater: bot.entity.isInWater,
    isSneaking: bot.entity.crouching || false,
    heldItem: bot.heldItem ? bot.heldItem.type : 0,
  };
}

function getFullState(bot) {
  return {
    player: getPlayerState(bot),
    entities: getVisibleEntities(bot),
  };
}

module.exports = {
  serializeEntity,
  getVisibleEntities,
  getPlayerState,
  getFullState,
  isHostile,
};
