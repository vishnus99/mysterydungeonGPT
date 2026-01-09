// Game state
let currentFloor = 0;
let totalFloors = 5;
let dungeonMaps = [];
let currentMap = null;
let player = { x: 0, y: 0 };
const TILE_SIZE = 12;

// Enemy state tracking
let enemies = []; // Array of enemy objects with position and state
let lastEnemyMoveTime = Date.now(); // Initialize to current time so enemies can move immediately
const ENEMY_MOVE_INTERVAL = 300; // Enemies move every 300ms (slower than player)

// Camera/viewport settings
const VIEWPORT_WIDTH_TILES = 6;  // Number of tiles visible horizontally (odd number for centering)
const VIEWPORT_HEIGHT_TILES = 3;  // Number of tiles visible vertically (odd number for centering)
const CANVAS_WIDTH = VIEWPORT_WIDTH_TILES * TILE_SIZE;
const CANVAS_HEIGHT = VIEWPORT_HEIGHT_TILES * TILE_SIZE;

let canvas, ctx;
let camera = { x: 0, y: 0 }; // Camera position in world coordinates

// Exploration tracking - Set of explored tile coordinates as "x,y" strings
let exploredTiles = new Set();
const EXPLORATION_RADIUS = 3; // Number of tiles around player that get explored (visibility range)

// Track currently pressed keys
const pressedKeys = new Set();

// Individual key directions
const KEY_DIRECTIONS = {
    'w': [0, -1], 'arrowup': [0, -1],
    's': [0, 1], 'arrowdown': [0, 1],
    'a': [-1, 0], 'arrowleft': [-1, 0],
    'd': [1, 0], 'arrowright': [1, 0]
};

// Initialize game
document.addEventListener('DOMContentLoaded', () => {
    canvas = document.getElementById('gameCanvas');
    ctx = canvas.getContext('2d');
    
    // Set fixed canvas size
    canvas.width = CANVAS_WIDTH;
    canvas.height = CANVAS_HEIGHT;
    
    document.getElementById('startButton').addEventListener('click', startGame);
    document.getElementById('resetButton').addEventListener('click', () => {
        document.getElementById('gameScreen').style.display = 'none';
        document.getElementById('menuScreen').style.display = 'block';
    });
    
    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('keyup', handleKeyUp);
    
    // Game loop for smooth movement and enemy AI
    setInterval(gameLoop, 100); // Update every 100ms (enemies move every 300ms internally)
});

async function startGame() {
    totalFloors = parseInt(document.getElementById('floorCount').value);
    
    const indexResponse = await fetch('maps/map_index.json');
    const index = await indexResponse.json();
    
    // Randomly select maps
    dungeonMaps = [...index.map_ids]
        .sort(() => Math.random() - 0.5)
        .slice(0, totalFloors);
    
    document.getElementById('menuScreen').style.display = 'none';
    document.getElementById('gameScreen').style.display = 'block';
    
    currentFloor = 0;
    loadFloor(0);
}

function updateCamera() {
    if (!currentMap) return;
    
    // Center camera on player
    // Camera position is the top-left corner of the viewport
    const targetCameraX = player.x - Math.floor(VIEWPORT_WIDTH_TILES / 2);
    const targetCameraY = player.y - Math.floor(VIEWPORT_HEIGHT_TILES / 2);
    
    // Clamp camera to map boundaries while keeping player as centered as possible
    if (currentMap.width <= VIEWPORT_WIDTH_TILES) {
        // Map is smaller than viewport - center the map
        camera.x = Math.floor((currentMap.width - VIEWPORT_WIDTH_TILES) / 2);
    } else {
        // Map is larger than viewport - clamp to boundaries
        camera.x = Math.max(0, Math.min(targetCameraX, currentMap.width - VIEWPORT_WIDTH_TILES));
    }
    
    if (currentMap.height <= VIEWPORT_HEIGHT_TILES) {
        // Map is smaller than viewport - center the map
        camera.y = Math.floor((currentMap.height - VIEWPORT_HEIGHT_TILES) / 2);
    } else {
        // Map is larger than viewport - clamp to boundaries
        camera.y = Math.max(0, Math.min(targetCameraY, currentMap.height - VIEWPORT_HEIGHT_TILES));
    }
}

function worldToScreen(worldX, worldY) {
    // Convert world coordinates to screen coordinates
    const screenX = (worldX - camera.x) * TILE_SIZE;
    const screenY = (worldY - camera.y) * TILE_SIZE;
    return [screenX, screenY];
}

function exploreTilesAround(x, y) {
    // Mark tiles within exploration radius as explored
    if (!currentMap) return;
    
    for (let dy = -EXPLORATION_RADIUS; dy <= EXPLORATION_RADIUS; dy++) {
        for (let dx = -EXPLORATION_RADIUS; dx <= EXPLORATION_RADIUS; dx++) {
            const exploreX = x + dx;
            const exploreY = y + dy;
            
            // Check bounds
            if (exploreX < 0 || exploreX >= currentMap.width ||
                exploreY < 0 || exploreY >= currentMap.height) {
                continue;
            }
            
            // Mark as explored (using Manhattan distance for diamond shape)
            const distance = Math.abs(dx) + Math.abs(dy);
            if (distance <= EXPLORATION_RADIUS) {
                exploredTiles.add(`${exploreX},${exploreY}`);
            }
        }
    }
}

function isExplored(x, y) {
    return exploredTiles.has(`${x},${y}`);
}

async function loadFloor(floorIndex) {
    if (floorIndex >= dungeonMaps.length) {
        alert('Congratulations! You completed the dungeon!');
        return;
    }
    
    const response = await fetch(`maps/${dungeonMaps[floorIndex]}.json`);
    currentMap = await response.json();
    
    player.x = currentMap.player_spawn[0];
    player.y = currentMap.player_spawn[1];
    
    // Initialize exploration tracking - start with spawn area explored
    exploredTiles = new Set();
    exploreTilesAround(player.x, player.y);
    
    // Initialize enemies with state tracking
    enemies = [];
    lastEnemyMoveTime = Date.now(); // Reset timer when loading new floor
    if (currentMap.enemies && Array.isArray(currentMap.enemies)) {
        currentMap.enemies.forEach(enemy => {
            const isOnFloor = isFloorTile(enemy.x, enemy.y);
            enemies.push({
                x: enemy.x,
                y: enemy.y,
                onFloor: isOnFloor, // Track if enemy has entered floor area
                trapped: isOnFloor  // If enemy starts on floor, they're immediately trapped
            });
        });
    }
    
    document.getElementById('currentFloor').textContent = floorIndex + 1;
    document.getElementById('totalFloors').textContent = totalFloors;
    
    // Canvas size is now fixed, no need to resize
    updateCamera();
    draw();
}

function isPositionOccupiedByEnemy(x, y, excludeEnemy = null) {
    // Check if any enemy (except the excluded one) is at this position
    return enemies.some(enemy => {
        if (excludeEnemy && enemy === excludeEnemy) return false;
        return enemy.x === x && enemy.y === y;
    });
}

function isPositionOccupiedByPlayer(x, y) {
    return player.x === x && player.y === y;
}

function isValidMove(x, y) {
    // Check bounds and walls
    if (x < 0 || x >= currentMap.width ||
        y < 0 || y >= currentMap.height ||
        currentMap.tiles[y][x] === 0) {
        return false;
    }
    
    // Players can now pass through enemies - removed enemy collision check
    
    return true;
}

function isFloorTile(x, y) {
    return x >= 0 && x < currentMap.width &&
           y >= 0 && y < currentMap.height &&
           currentMap.tiles[y][x] === 1; // 1 = floor tile
}

function isValidEnemyMove(enemy, newX, newY) {
    // Basic bounds check
    if (newX < 0 || newX >= currentMap.width || 
        newY < 0 || newY >= currentMap.height) {
        return false;
    }
    
    // Enemies can now move through walls - removed wall check
    
    // Check collision with player
    if (isPositionOccupiedByPlayer(newX, newY)) {
        return false;
    }
    
    // Check collision with other enemies
    if (isPositionOccupiedByEnemy(newX, newY, enemy)) {
        return false;
    }
    
    // Enemies can move through walls regardless of trapped state
    return true;
}

function moveEnemyTowardsPlayer(enemy) {
    if (!currentMap) return;
    
    // Calculate direction towards player
    const dx = player.x - enemy.x;
    const dy = player.y - enemy.y;
    
    // If enemy is already at player position, don't move
    if (dx === 0 && dy === 0) {
        return;
    }
    
    // Try 8-directional movement, prioritizing direction towards player
    const directions = [
        [Math.sign(dx), 0],           // Horizontal towards player
        [0, Math.sign(dy)],           // Vertical towards player
        [Math.sign(dx), Math.sign(dy)], // Diagonal towards player
        [1, 0], [-1, 0], [0, 1], [0, -1],  // Cardinal
        [1, 1], [1, -1], [-1, 1], [-1, -1] // Diagonal
    ];
    
    // Remove duplicates and zero movements
    const uniqueDirections = [];
    const seen = new Set();
    for (const [dx, dy] of directions) {
        if (dx === 0 && dy === 0) continue;
        const key = `${dx},${dy}`;
        if (!seen.has(key)) {
            seen.add(key);
            uniqueDirections.push([dx, dy]);
        }
    }
    
    // Sort directions by distance to player (closer first)
    uniqueDirections.sort((a, b) => {
        const distA = Math.abs((enemy.x + a[0]) - player.x) + Math.abs((enemy.y + a[1]) - player.y);
        const distB = Math.abs((enemy.x + b[0]) - player.x) + Math.abs((enemy.y + b[1]) - player.y);
        return distA - distB;
    });
    
    // Try each direction
    for (const [moveX, moveY] of uniqueDirections) {
        const newX = enemy.x + moveX;
        const newY = enemy.y + moveY;
        
        if (isValidEnemyMove(enemy, newX, newY)) {
            // Check if enemy is entering floor area
            if (isFloorTile(newX, newY) && !enemy.onFloor) {
                enemy.onFloor = true;
                enemy.trapped = true; // Once on floor, they're trapped
            }
            
            // Update enemy position
            enemy.x = newX;
            enemy.y = newY;
            return; // Successfully moved
        }
    }
    
    // If no valid move found, enemy stays in place
}

function updateEnemies() {
    if (!currentMap || enemies.length === 0) {
        return; // No map or no enemies to update
    }
    
    const currentTime = Date.now();
    
    // Only move enemies at intervals (slower than player movement)
    if (currentTime - lastEnemyMoveTime < ENEMY_MOVE_INTERVAL) {
        return;
    }
    
    lastEnemyMoveTime = currentTime;
    
    // Move each enemy towards player
    enemies.forEach(enemy => {
        moveEnemyTowardsPlayer(enemy);
    });
}

function checkStairs() {
    if (player.x === currentMap.stairs_spawn[0] && 
        player.y === currentMap.stairs_spawn[1]) {
        loadFloor(++currentFloor);
        return true;
    }
    return false;
}

function getCombinedDirection() {
    let dx = 0;
    let dy = 0;
    
    // Combine all pressed keys
    for (const key of pressedKeys) {
        const direction = KEY_DIRECTIONS[key.toLowerCase()];
        if (direction) {
            dx += direction[0];
            dy += direction[1];
        }
    }
    
    // Normalize diagonal movement (limit to 1 step per axis)
    if (dx !== 0) dx = dx > 0 ? 1 : -1;
    if (dy !== 0) dy = dy > 0 ? 1 : -1;
    
    return [dx, dy];
}

function gameLoop() {
    if (!currentMap) return;
    
    let playerMoved = false;
    
    // Handle player movement
    if (pressedKeys.size > 0) {
        const [dx, dy] = getCombinedDirection();
        if (dx !== 0 || dy !== 0) {
            const newX = player.x + dx;
            const newY = player.y + dy;
            
            if (isValidMove(newX, newY)) {
                player.x = newX;
                player.y = newY;
                playerMoved = true;
                
                // Explore tiles around new player position
                exploreTilesAround(player.x, player.y);
                
                updateCamera(); // Update camera position
                
                if (checkStairs()) {
                    return; // Floor changed, don't draw
                }
            }
        }
    }
    
    // Update enemies (they move independently of player input)
    updateEnemies();
    
    // Always redraw to show enemy movement, even if player didn't move
    draw();
}

function handleKeyDown(e) {
    if (!currentMap) return;
    
    const key = e.key.toLowerCase();
    if (KEY_DIRECTIONS[key]) {
        pressedKeys.add(key);
        e.preventDefault();
    }
}

function handleKeyUp(e) {
    const key = e.key.toLowerCase();
    pressedKeys.delete(key);
    e.preventDefault();
}

function draw() {
    if (!currentMap) return;
    
    // Clear and draw background
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Calculate visible tile range
    const startX = Math.max(0, camera.x);
    const endX = Math.min(currentMap.width, camera.x + VIEWPORT_WIDTH_TILES);
    const startY = Math.max(0, camera.y);
    const endY = Math.min(currentMap.height, camera.y + VIEWPORT_HEIGHT_TILES);
    
    // Draw visible tiles only (only show explored tiles)
    for (let y = startY; y < endY; y++) {
        for (let x = startX; x < endX; x++) {
            const [screenX, screenY] = worldToScreen(x, y);
            
            if (isExplored(x, y)) {
                // Draw explored tiles
                const tile = currentMap.tiles[y][x];
                if (tile === 0) {
                    ctx.fillStyle = '#333';
                    ctx.fillRect(screenX, screenY, TILE_SIZE, TILE_SIZE);
                } else {
                    ctx.fillStyle = '#8B4513';
                    ctx.fillRect(screenX, screenY, TILE_SIZE, TILE_SIZE);
                }
            } else {
                // Draw unexplored tiles as black
                ctx.fillStyle = '#000';
                ctx.fillRect(screenX, screenY, TILE_SIZE, TILE_SIZE);
            }
        }
    }
    
    // Draw enemies (use the enemies array with state tracking) - only if explored
    enemies.forEach(enemy => {
        if (enemy.x >= startX && enemy.x < endX && 
            enemy.y >= startY && enemy.y < endY &&
            isExplored(enemy.x, enemy.y)) {
            const [screenEx, screenEy] = worldToScreen(enemy.x, enemy.y);
            
            // Color based on enemy type
            // Simple enemy rendering - just red color
            const enemyColor = '#FF6B6B'; // Red for all enemies
            
            // Visual indicator if enemy is trapped on floor
            if (enemy.trapped) {
                ctx.fillStyle = enemyColor;
                ctx.fillRect(screenEx, screenEy, TILE_SIZE, TILE_SIZE);
                // Add a border to indicate trapped state
                ctx.strokeStyle = '#FFFF00'; // Yellow border for trapped enemies
                ctx.lineWidth = 1;
                ctx.strokeRect(screenEx, screenEy, TILE_SIZE, TILE_SIZE);
            } else {
                ctx.fillStyle = enemyColor;
                ctx.fillRect(screenEx, screenEy, TILE_SIZE, TILE_SIZE);
            }
            
            ctx.fillStyle = '#fff';
            ctx.font = `${TILE_SIZE}px Arial`;
            ctx.textAlign = 'center';
            ctx.fillText('E', screenEx + TILE_SIZE/2, screenEy + TILE_SIZE/2 + 3);
        }
    });
    
    // Draw stairs (if visible and explored)
    const [sx, sy] = currentMap.stairs_spawn;
    if (sx >= startX && sx < endX && sy >= startY && sy < endY &&
        isExplored(sx, sy)) {
        const [screenSx, screenSy] = worldToScreen(sx, sy);
        ctx.fillStyle = '#ff0000';
        ctx.fillRect(screenSx, screenSy, TILE_SIZE, TILE_SIZE);
        ctx.fillStyle = '#fff';
        ctx.font = `${TILE_SIZE}px Arial`;
        ctx.textAlign = 'center';
        ctx.fillText('↓', screenSx + TILE_SIZE/2, screenSy + TILE_SIZE/2 + 3);
    }
    
    // Draw player (always visible since camera follows player)
    const [screenPx, screenPy] = worldToScreen(player.x, player.y);
    ctx.fillStyle = '#00ff00';
    ctx.fillRect(screenPx, screenPy, TILE_SIZE, TILE_SIZE);
    ctx.fillStyle = '#fff';
    ctx.fillText('@', screenPx + TILE_SIZE/2, screenPy + TILE_SIZE/2 + 3);
}