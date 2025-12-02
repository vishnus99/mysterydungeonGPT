// Game state
let currentFloor = 0;
let totalFloors = 5;
let dungeonMaps = [];
let currentMap = null;
let player = { x: 0, y: 0 };
const TILE_SIZE = 12;

// Camera/viewport settings
const VIEWPORT_WIDTH_TILES = 50;  // Number of tiles visible horizontally
const VIEWPORT_HEIGHT_TILES = 40; // Number of tiles visible vertically
const CANVAS_WIDTH = VIEWPORT_WIDTH_TILES * TILE_SIZE;
const CANVAS_HEIGHT = VIEWPORT_HEIGHT_TILES * TILE_SIZE;

let canvas, ctx;
let camera = { x: 0, y: 0 }; // Camera position in world coordinates

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
    
    // Game loop for smooth movement
    setInterval(gameLoop, 100); // Update every 100ms
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
    camera.x = player.x - Math.floor(VIEWPORT_WIDTH_TILES / 2);
    camera.y = player.y - Math.floor(VIEWPORT_HEIGHT_TILES / 2);
    
    // Clamp camera to map boundaries
    camera.x = Math.max(0, Math.min(camera.x, currentMap.width - VIEWPORT_WIDTH_TILES));
    camera.y = Math.max(0, Math.min(camera.y, currentMap.height - VIEWPORT_HEIGHT_TILES));
}

function worldToScreen(worldX, worldY) {
    // Convert world coordinates to screen coordinates
    const screenX = (worldX - camera.x) * TILE_SIZE;
    const screenY = (worldY - camera.y) * TILE_SIZE;
    return [screenX, screenY];
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
    
    document.getElementById('currentFloor').textContent = floorIndex + 1;
    document.getElementById('totalFloors').textContent = totalFloors;
    
    // Canvas size is now fixed, no need to resize
    updateCamera();
    draw();
}

function isValidMove(x, y) {
    return x >= 0 && x < currentMap.width &&
           y >= 0 && y < currentMap.height &&
           currentMap.tiles[y][x] !== 0;
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
    if (!currentMap || pressedKeys.size === 0) return;
    
    const [dx, dy] = getCombinedDirection();
    if (dx === 0 && dy === 0) return;
    
    const newX = player.x + dx;
    const newY = player.y + dy;
    
    if (isValidMove(newX, newY)) {
        player.x = newX;
        player.y = newY;
        
        updateCamera(); // Update camera position
        
        if (!checkStairs()) {
            draw();
        }
    }
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
    
    // Draw visible tiles only
    for (let y = startY; y < endY; y++) {
        for (let x = startX; x < endX; x++) {
            const tile = currentMap.tiles[y][x];
            const [screenX, screenY] = worldToScreen(x, y);
            
            if (tile === 0) {
                ctx.fillStyle = '#333';
                ctx.fillRect(screenX, screenY, TILE_SIZE, TILE_SIZE);
            } else {
                ctx.fillStyle = '#8B4513';
                ctx.fillRect(screenX, screenY, TILE_SIZE, TILE_SIZE);
            }
        }
    }
    
    // Draw stairs (if visible)
    const [sx, sy] = currentMap.stairs_spawn;
    if (sx >= startX && sx < endX && sy >= startY && sy < endY) {
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