// Game state
let currentFloor = 0;
let totalFloors = 5;
let dungeonMaps = [];
let currentMap = null;
let player = { x: 0, y: 0 };
const TILE_SIZE = 12;
let canvas, ctx;

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
    
    canvas.width = currentMap.width * TILE_SIZE;
    canvas.height = currentMap.height * TILE_SIZE;
    
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
    
    // Draw tiles
    for (let y = 0; y < currentMap.height; y++) {
        for (let x = 0; x < currentMap.width; x++) {
            const tile = currentMap.tiles[y][x];
            const screenX = x * TILE_SIZE;
            const screenY = y * TILE_SIZE;
            
            if (tile === 0) {
                ctx.fillStyle = '#333';
                ctx.fillRect(screenX, screenY, TILE_SIZE, TILE_SIZE);
            } else {
                ctx.fillStyle = '#8B4513';
                ctx.fillRect(screenX, screenY, TILE_SIZE, TILE_SIZE);
            }
        }
    }
    
    // Draw stairs
    const [sx, sy] = currentMap.stairs_spawn;
    ctx.fillStyle = '#ff0000';
    ctx.fillRect(sx * TILE_SIZE, sy * TILE_SIZE, TILE_SIZE, TILE_SIZE);
    ctx.fillStyle = '#fff';
    ctx.font = `${TILE_SIZE}px Arial`;
    ctx.textAlign = 'center';
    ctx.fillText('↓', sx * TILE_SIZE + TILE_SIZE/2, sy * TILE_SIZE + TILE_SIZE/2 + 3);
    
    // Draw player
    ctx.fillStyle = '#00ff00';
    ctx.fillRect(player.x * TILE_SIZE, player.y * TILE_SIZE, TILE_SIZE, TILE_SIZE);
    ctx.fillStyle = '#fff';
    ctx.fillText('@', player.x * TILE_SIZE + TILE_SIZE/2, player.y * TILE_SIZE + TILE_SIZE/2 + 3);
}

