import * as THREE from 'three';
import {STLLoader} from 'three/addons/loaders/STLLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';


const hoiRenderWindow = document.getElementById("js-container");
const windowHeight = 0.5 * hoiRenderWindow.offsetWidth;
const windowWidth = hoiRenderWindow.offsetWidth;

const renderer = new THREE.WebGLRenderer();
renderer.setSize(windowWidth, windowHeight);
renderer.setClearColor(0xffffff,1.0);
renderer.shadowMap.enabled = true;
document.getElementById("js-container").appendChild(renderer.domElement);

const camera = new THREE.PerspectiveCamera(45, windowWidth / windowHeight, 0.01, 500)
camera.position.set(1, 1, 5);
camera.lookAt(0, 0, 0);


const scene = new THREE.Scene();

var loader = new STLLoader();

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.damplingFacor = 0.01;
controls.autoRotate = true;

function animate() {

	controls.update();

	renderer.render( scene, camera );

}
renderer.setAnimationLoop( animate );

const SMPLBoneIndices =  [[ 0,  1], [ 0,  2], [ 0,  3], 
                [ 1,  4], [ 2,  5], [ 3,  6], 
                [ 4,  7], [ 5,  8], [ 6,  9], 
                [ 7, 10], [ 8, 11], [ 9, 12], 
                [ 9, 13], [ 9, 14], [12, 15],
                [13, 16], [14, 17], [16, 18],
                [17, 19], [18, 20], [19, 21]];


function resetCamera() {
	camera.position.set(1, 1, 5);
	camera.lookAt(0, 0, 0);
}

function clearScene() {
	scene.traverse((child) => {
		if (child.material) {
			child.material.dispose();
		}
		if (child.geometry) {
			child.geometry.dispose();
		}
		child = null;
    });
    scene.clear();
}


function drawScene(smpl_mesh_fie, object_mesh_file, kps_file, objectBoneIndices, xyzRange, objectKpsNum, plane_y) {

	const planeGeometry = new THREE.PlaneGeometry(5, 5);
	const material = new THREE.MeshStandardMaterial({color: 0xFFFFFF, roughness: 0.5 });
	const plane = new THREE.Mesh(planeGeometry, material);
	plane.position.set(0, plane_y, 0);
	plane.rotation.x = - 3.14159 / 2;
	plane.receiveShadow = true;
	scene.add(plane);

	const ambientLight = new THREE.AmbientLight(0xffffff, 2);
	scene.add(ambientLight);
	const directionalLight = new THREE.DirectionalLight(0xffffff, 5.);
	directionalLight.position.set(3, 5, 5);
	directionalLight.castShadow = true;
	scene.add(directionalLight);

	loader.load(smpl_mesh_fie, function (geometry) {
		const material = new THREE.MeshStandardMaterial({metalness:0.1, roughness: 0.8,  color: 0xA9A9A9 });
		const mesh = new THREE.Mesh(geometry, material);
		mesh.castShadow = true;
		scene.add(mesh);
	})

	loader.load(object_mesh_file, function (geometry) {
		const material = new THREE.MeshStandardMaterial({metalness:0.1, roughness: 0.8,  color: 0x696969 });
		const mesh = new THREE.Mesh(geometry, material);
		mesh.castShadow = true;
		scene.add(mesh);
	})

	const fileLoader = new THREE.FileLoader();
	fileLoader.load(
		kps_file,

		function (data) {
			const arrayOfLines = data.split('\n').map(line => line.trim());
			arrayOfLines.forEach((line, index) => {
				const keypoints = line.split(' ').map(item => parseFloat(item.trim()));
				const points = [];
				points.push(new THREE.Vector3(keypoints[0 * 3], keypoints[0 * 3 + 1], keypoints[0 * 3 + 2]));
				points.push(new THREE.Vector3(keypoints[1 * 3], keypoints[1 * 3 + 1], keypoints[1 * 3 + 2]));
				points.push(new THREE.Vector3(keypoints[0 * 3], keypoints[0 * 3 + 1], keypoints[0 * 3 + 2]));
				points.push(new THREE.Vector3(keypoints[2 * 3], keypoints[2 * 3 + 1], keypoints[2 * 3 + 2]));
				points.push(new THREE.Vector3(keypoints[0 * 3], keypoints[0 * 3 + 1], keypoints[0 * 3 + 2]));
				points.push(new THREE.Vector3(keypoints[3 * 3], keypoints[3 * 3 + 1], keypoints[3 * 3 + 2]));
				points.push(new THREE.Vector3(keypoints[0 * 3], keypoints[0 * 3 + 1], keypoints[0 * 3 + 2]));
				points.push(new THREE.Vector3(keypoints[4 * 3], keypoints[4 * 3 + 1], keypoints[4 * 3 + 2]));

				points.push(new THREE.Vector3(keypoints[1 * 3], keypoints[1 * 3 + 1], keypoints[1 * 3 + 2]));
				points.push(new THREE.Vector3(keypoints[2 * 3], keypoints[2 * 3 + 1], keypoints[2 * 3 + 2]));
				points.push(new THREE.Vector3(keypoints[3 * 3], keypoints[3 * 3 + 1], keypoints[3 * 3 + 2]));
				points.push(new THREE.Vector3(keypoints[4 * 3], keypoints[4 * 3 + 1], keypoints[4 * 3 + 2]));
				points.push(new THREE.Vector3(keypoints[1 * 3], keypoints[1 * 3 + 1], keypoints[1 * 3 + 2]));

				const geometry = new THREE.BufferGeometry().setFromPoints(points);
				const material = new THREE.LineBasicMaterial( { color: 0x000000 } );
				const lines = new THREE.Line(geometry, material);
				scene.add(lines);

				var xMax = xyzRange[0];
				var xMin = xyzRange[1];
				var yMax = xyzRange[2];
				var yMin = xyzRange[3];
				var zMax = xyzRange[4];
				var zMin = xyzRange[5];

				SMPLBoneIndices.forEach((item, index) => {
					const SMPLPoints = []
					SMPLPoints.push(new THREE.Vector3(keypoints[(item[0] + 5) * 3], keypoints[(item[0] + 5) * 3 + 1], keypoints[(item[0] + 5) * 3 + 2]));
					SMPLPoints.push(new THREE.Vector3(keypoints[(item[1] + 5) * 3], keypoints[(item[1] + 5) * 3 + 1], keypoints[(item[1] + 5) * 3 + 2]));

					var x = keypoints[(item[0] + 5) * 3];
					var y = keypoints[(item[0] + 5) * 3 + 1];
					var z = keypoints[(item[0] + 5) * 3 + 2];
					var r = parseInt(255 * (x - xMin) / (xMax - xMin));
					var g = parseInt(255 * (y - yMin) / (yMax - yMin));
					var b = parseInt(255 * (z - zMin) / (zMax - zMin));

					const SMPLGeometry = new THREE.BufferGeometry().setFromPoints(SMPLPoints);
					const SMPLMaterial = new THREE.LineBasicMaterial( { color: `rgb(${r}, ${g}, ${b})` } );
					const SMPLLines = new THREE.Line(SMPLGeometry, SMPLMaterial);
					scene.add(SMPLLines);
				});

				objectBoneIndices.forEach((item, index) => {
					const ObjectPoints = []
					ObjectPoints.push(new THREE.Vector3(keypoints[(item[0] + 27) * 3], keypoints[(item[0] + 27) * 3 + 1], keypoints[(item[0] + 27) * 3 + 2]));
					ObjectPoints.push(new THREE.Vector3(keypoints[(item[1] + 27) * 3], keypoints[(item[1] + 27) * 3 + 1], keypoints[(item[1] + 27) * 3 + 2]));

					var x = keypoints[(item[0] + 27) * 3];
					var y = keypoints[(item[0] + 27) * 3 + 1];
					var z = keypoints[(item[0] + 27) * 3 + 2];
					var r = parseInt(255 * (x - xMin) / (xMax - xMin));
					var g = parseInt(255 * (y - yMin) / (yMax - yMin));
					var b = parseInt(255 * (z - zMin) / (zMax - zMin));

					const ObjectGeometry = new THREE.BufferGeometry().setFromPoints(ObjectPoints);
					const ObjectMaterial = new THREE.LineBasicMaterial( { color: `rgb(${r}, ${g}, ${b})` } );
					const ObjectLines = new THREE.Line(ObjectGeometry, ObjectMaterial);
					scene.add(ObjectLines);
				});

				for (var i = 0; i < 22 + objectKpsNum; i++) {
					const points2 = [];
					points2.push(new THREE.Vector3(keypoints[(5 + i) * 3], keypoints[(5 + i) * 3 + 1], keypoints[(5 + i) * 3 + 2]));
					var x = keypoints[(5 + i) * 3];
					var y = keypoints[(5 + i) * 3 + 1];
					var z = keypoints[(5 + i) * 3 + 2];
					var r = parseInt(255 * (x - xMin) / (xMax - xMin));
					var g = parseInt(255 * (y - yMin) / (yMax - yMin));
					var b = parseInt(255 * (z - zMin) / (zMax - zMin));
					const geometry2 = new THREE.BufferGeometry().setFromPoints(points2);
					const material2 = new THREE.PointsMaterial({color: `rgb(${r}, ${g}, ${b})`, size: 0.02});
					const pts = new THREE.Points(geometry2, material2);
					scene.add(pts);
				}
			});
		}
	)
}

const barbellBoneIndices = [[0, 1], [1, 2], [2, 3], ];
const baseballBoneIndices = [[0, 1]];
const basketballBoneIndices = [[0, 0]];
const bicycleBoneIndices = [[0, 5], [1, 5], [2, 4], [3, 4], [4, 5], [4, 8], [5, 8], [8, 9], [6, 9], [7, 9]];
const celloBoneIndices = [[0, 1], [1, 2], [2, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 7], 
          [2, 8], [3, 9], [4, 10], [5, 11], [6, 12], [7, 13], 
          [8, 9], [8, 10], [9, 11], [10, 12], [11, 13], [12, 13]];
const skateboardBoneIndices = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 0]];
const tennisBoneIndices = [[0, 1], [1, 2], [1, 3], [2, 3], [2, 4], [4, 6], [3, 5], [5, 6]];
const violineBoneIndices = [[0, 1], [1, 2], [2, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 7], 
          [2, 8], [3, 9], [4, 10], [5, 11], [6, 12], [7, 13], 
          [8, 9], [8, 10], [9, 11], [10, 12], [11, 13], [12, 13]];

const barbellXYZRange = [2.76, -2.67, 2.98, -2.77, 3.60, -2.54];
const baseballXYZRange = [4.25, -4.35, 1.77, -0.98, 4.19, -3.74];
const basketballXYZRange = [4.04, -4.41, 0.90, -0.79, 4.20, -4.07];
const bicycleXYZRange = [3.58, -3.26, 1.83, -1.46, 3.62, -3.57];
const cellXYZRange = [3.07, -3.54, 1.44, -2.23, 3.81, -1.56];
const skateboardXYZRange = [4.52, -4.02, 2.75, -2.54, 4.38, -4.30];
const tennisXYZRange = [4.52, -4.02, 2.75, -2.54, 4.38, -4.30];
const violinXYZRange = [3.36, -4.11, 1.48, -1.33, 4.46, -2.57];

drawScene('static/barbell_smpl_mesh.stl', 'static/barbell_mesh.stl', 'static/barbell_kps.txt', barbellBoneIndices, barbellXYZRange, 4, -0.6);


document.getElementById("barbell").addEventListener("click", function (e) {
	if (e.button == 0) {
		clearScene();
		resetCamera();
		drawScene('static/barbell_smpl_mesh.stl', 'static/barbell_mesh.stl', 'static/barbell_kps.txt', barbellBoneIndices, barbellXYZRange, 4, -0.6);
	}
})

document.getElementById("baseball").addEventListener("click", function (e) {
	if (e.button == 0) {
		clearScene();
		resetCamera();
		drawScene('static/baseball_smpl_mesh.stl', 'static/baseball_mesh.stl', 'static/baseball_kps.txt', baseballBoneIndices, baseballXYZRange, 2, -0.9);
	}
})

document.getElementById("basketball").addEventListener("click", function (e) {
	if (e.button == 0) {
		clearScene();
		resetCamera();
		drawScene('static/basketball_smpl_mesh.stl', 'static/basketball_mesh.stl', 'static/basketball_kps.txt', basketballBoneIndices, basketballXYZRange, 1, -1);
	}
})

document.getElementById("bicycle").addEventListener("click", function (e) {
	if (e.button == 0) {
		clearScene();
		resetCamera();
		drawScene('static/bicycle_smpl_mesh.stl', 'static/bicycle_mesh.stl', 'static/bicycle_kps.txt', bicycleBoneIndices, bicycleXYZRange, 10, -1.3);
	}
})

document.getElementById("cello").addEventListener("click", function (e) {
	if (e.button == 0) {
		clearScene();
		resetCamera();
		drawScene('static/cello_smpl_mesh.stl', 'static/cello_mesh.stl', 'static/cello_kps.txt', celloBoneIndices, cellXYZRange, 14, -1);
	}
})

document.getElementById("skateboard").addEventListener("click", function (e) {
	if (e.button == 0) {
		clearScene();
		resetCamera();
		drawScene('static/skateboard_smpl_mesh.stl', 'static/skateboard_mesh.stl', 'static/skateboard_kps.txt', skateboardBoneIndices, skateboardXYZRange, 8, -1);
	}
})

document.getElementById("tennis").addEventListener("click", function (e) {
	if (e.button == 0) {
		clearScene();
		resetCamera();
		drawScene('static/tennis_smpl_mesh.stl', 'static/tennis_mesh.stl', 'static/tennis_kps.txt', tennisBoneIndices, tennisXYZRange, 7, -1);
	}
})

document.getElementById("violin").addEventListener("click", function (e) {
	if (e.button == 0) {
		clearScene();
		resetCamera();
		drawScene('static/violin_smpl_mesh.stl', 'static/violin_mesh.stl', 'static/violin_kps.txt', violineBoneIndices, violinXYZRange, 14, -1);
	}
})
