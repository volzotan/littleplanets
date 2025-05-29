use chrono::Utc;
use image::{DynamicImage, EncodableLayout, GenericImage, GenericImageView, GrayImage, ImageBuffer, ImageReader, Rgb, RgbImage};
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs::File;
use std::io::{BufWriter, Write};
use rstar::primitives::GeomWithData;
use rstar::RTree;
use std::f32::consts::{PI, TAU};

fn midpoint(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) / 2.0,
        a[1] + (b[1] - a[1]) / 2.0,
        a[2] + (b[2] - a[2]) / 2.0,
    ]
}
struct Mesh {
    points: Vec<[f32; 3]>,
    triangles: Vec<[usize; 3]>,
    colors: Vec<[f32; 3]>,
}

impl<'a> Mesh {
    fn new(points: Vec<[f32; 3]>, triangles: Vec<[usize; 3]>, colors: Vec<[f32; 3]>) -> Self {
        Mesh {
            points,
            triangles,
            colors,
        }
    }

    fn fill_color(&mut self, color: [f32; 3]) {
        self.colors = vec![color; self.triangles.len()];
    }

    fn add_color(&mut self, raster: &RgbImage) {
        for i in 0..self.triangles.len() {
            let p = self.triangle_center(i);
            let d = (p[0].powi(2) + p[1].powi(2) + p[2].powi(2)).sqrt();

            let lat = (p[2] / d).acos();
            let lon = p[1].atan2(p[0]);

            let y = (lat / PI * raster.height() as f32) as u32;
            let x = (((lon / TAU) + 0.5) * (raster.width()-1) as f32) as u32;
            let pixel = raster.get_pixel(x, y);

            self.colors[i] = [
                pixel[0] as f32 / 255.0,
                pixel[1] as f32 / 255.0,
                pixel[2] as f32 / 255.0
            ];
        }
    }

    fn index_to_key(mut indices: [usize; 2]) -> String {
        indices.sort();
        format!["{}|{}", indices[0], indices[1]]
    }

    // fn hash_points(&mut self, mut map_points: &HashMap<String, usize>, key: String, point: [f32; 3]) -> usize {
    //     match map_points.get(&key) {
    //         Some(ind) => {
    //             *ind
    //         }
    //         None => {
    //             self.points.push(point);
    //             let ind = self.points.len() - 1;
    //             map_points.insert(key, ind);
    //             ind
    //         }
    //     }
    // }

    fn subdivide(&mut self, iterations: u32) -> () {
        for _ in 0..iterations {
            let mut subdivision_triangles: Vec<[usize; 3]> = vec![];
            let mut map_points: HashMap<String, usize> = HashMap::new();

            for t in self.triangles.iter() {
                let i0 = t[0];
                let i1 = t[1];
                let i2 = t[2];

                let alpha = midpoint(self.points[i0], self.points[i1]);
                let beta = midpoint(self.points[i1], self.points[i2]);
                let gamma = midpoint(self.points[i2], self.points[i0]);

                let k_alpha = Self::index_to_key([i0, i1]);
                let k_beta = Self::index_to_key([i1, i2]);
                let k_gamma = Self::index_to_key([i2, i0]);

                let i_alpha;
                match map_points.get(&k_alpha) {
                    Some(ind) => {
                        i_alpha = *ind;
                    }
                    None => {
                        self.points.push(alpha);
                        i_alpha = self.points.len() - 1;
                        map_points.insert(k_alpha, i_alpha);
                    }
                }

                let i_beta;
                match map_points.get(&k_beta) {
                    Some(ind) => {
                        i_beta = *ind;
                    }
                    None => {
                        self.points.push(beta);
                        i_beta = self.points.len() - 1;
                        map_points.insert(k_beta, i_beta);
                    }
                }

                let i_gamma;
                match map_points.get(&k_gamma) {
                    Some(ind) => {
                        i_gamma = *ind;
                    }
                    None => {
                        self.points.push(gamma);
                        i_gamma = self.points.len() - 1;
                        map_points.insert(k_gamma, i_gamma);
                    }
                }

                subdivision_triangles.push([i0, i_alpha, i_gamma]);
                subdivision_triangles.push([i_alpha, i1, i_beta]);
                subdivision_triangles.push([i_beta, i2, i_gamma]);
                subdivision_triangles.push([i_gamma, i_alpha, i_beta]);
            }

            self.triangles = subdivision_triangles;
        }
        self.colors = vec![[1.0, 1.0, 1.0]; self.triangles.len()]
    }

    fn project_to_sphere(&mut self) {
        for i in 0..self.points.len() {
            let p = self.points[i];
            let d = (p[0].powi(2) + p[1].powi(2) + p[2].powi(2)).sqrt();

            let lat = (p[2] / d).acos();
            let lon = p[1].atan2(p[0]);

            let r = 1.0;
            let x = r * lat.sin() * lon.cos();
            let y = r * lat.sin() * lon.sin();
            let z = r * lat.cos();

            self.points[i] = [x, y, z];
        }
    }

    fn triangle_center(&self, triangle_index: usize) -> [f32; 3] {
        let t = self.triangles[triangle_index];
        [
            (self.points[t[0]][0] + self.points[t[1]][0] + self.points[t[2]][0]) / 3.0,
            (self.points[t[0]][1] + self.points[t[1]][1] + self.points[t[2]][1]) / 3.0,
            (self.points[t[0]][2] + self.points[t[1]][2] + self.points[t[2]][2]) / 3.0
        ]
    }
    fn add_seedpoints(&mut self, num: u32) {

        type PointIndex = GeomWithData<[f32; 3], usize>;

        let mut points: Vec<PointIndex> = Vec::with_capacity(self.triangles.len());
        for i in 0..self.triangles.len() {
            points.push(PointIndex::new(self.triangle_center(i), i));
        }
        let mut tree = RTree::bulk_load(points);

        for i in 0..num {
            let phi = (1.0 - 2.0 * (i as f32 + 0.5) / num as f32).acos();
            let theta =  PI * (1.0 + 5.0_f32.powf(0.5)) * (i as f32 + 0.5);

            let x = theta.cos() * phi.sin();
            let y = theta.sin() * phi.sin();
            let z = phi.cos();

            match tree.nearest_neighbor(&[x, y, z]) {
                Some(point_index) => {
                    self.colors[point_index.data] = [1.0, 0.0, 0.0];
                }
                None => {}
            }
        }
    }

    fn get_seedpoints(&mut self, num: u32) -> Vec<[f32; 3]> {

        type PointIndex = GeomWithData<[f32; 3], usize>;

        let mut center_points: Vec<PointIndex> = Vec::with_capacity(self.triangles.len());
        for i in 0..self.triangles.len() {
            center_points.push(PointIndex::new(self.triangle_center(i), i));
        }
        let mut tree = RTree::bulk_load(center_points);

        let mut seedpoints: Vec<[f32; 3]> = Vec::with_capacity(self.triangles.len());
        for i in 0..num {
            let phi = (1.0 - 2.0 * (i as f32 + 0.5) / num as f32).acos();
            let theta =  PI * (1.0 + 5.0_f32.powf(0.5)) * (i as f32 + 0.5);

            let x = theta.cos() * phi.sin();
            let y = theta.sin() * phi.sin();
            let z = phi.cos();

            match tree.nearest_neighbor(&[x, y, z]) {
                Some(point_index) => {
                    seedpoints.push(point_index.geom().clone());
                }
                None => {}
            }
        }
        seedpoints
    }

    // fn project(&mut self, raster_height: &GrayImage, raster_color: &RgbImage, scale: f32) {
    //     for i in 0..self.points.len() {
    //         let p = self.points[i];
    //         let d = (p[0].powi(2) + p[1].powi(2) + p[2].powi(2)).sqrt();
    //
    //         let lat = (p[2] / d).acos();
    //         let lon = p[1].atan2(p[0]);
    //
    //         let (px, py) = Mesh::map(raster_height.width(), raster_height(), lat, lon);
    //         let r = 1.0 + (raster_height.get_pixel(py, py)[0] as f32) / 255.0 * scale;
    //
    //         let x = r * lat.sin() * lon.cos();
    //         let y = r * lat.sin() * lon.sin();
    //         let z = r * lat.cos();
    //
    //         self.points[i] = [x, y, z];
    //         let c = map()
    //     }
    // }

    fn write_ply_with_vertex_colors(
        &self,
        filename: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let header = format!(
            "ply
            format ascii 1.0
            element vertex {num_points}
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            element face {num_triangles}
            property list uchar int vertex_indices
            end_header",
            num_points = self.points.len(),
            num_triangles = self.triangles.len()
        );

        let file = File::create(filename)?;
        let mut writer = BufWriter::new(&file);

        for line in header.lines() {
            writer.write(line.trim().as_bytes())?;
            writer.write("\n".as_bytes())?;
        }

        for i in 0..self.points.len() {
            writer.write_fmt(format_args!(
                "{:.4} {:.4} {:.4} {} {} {}\n",
                self.points[i][0],
                self.points[i][1],
                self.points[i][2],
                (self.colors[i][0] * 255.0) as u32,
                (self.colors[i][1] * 255.0) as u32,
                (self.colors[i][2] * 255.0) as u32,
            ))?;
        }

        for i in 0..self.triangles.len() {
            writer.write_fmt(format_args!(
                "3 {} {} {}\n",
                self.triangles[i][0], self.triangles[i][1], self.triangles[i][2],
            ))?;
        }

        Ok(())
    }

    fn write_ply_with_triangle_colors(
        &self,
        filename: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let header = format!(
            "ply
            format ascii 1.0
            element vertex {num_points}
            property float x
            property float y
            property float z
            element face {num_triangles}
            property list uchar int vertex_indices
            property uchar red
            property uchar green
            property uchar blue
            end_header",
            num_points = self.points.len(),
            num_triangles = self.triangles.len()
        );

        let file = File::create(filename)?;
        let mut writer = BufWriter::new(&file);

        for line in header.lines() {
            writer.write(line.trim().as_bytes())?;
            writer.write("\n".as_bytes())?;
        }

        for i in 0..self.points.len() {
            writer.write_fmt(format_args!(
                "{:.4} {:.4} {:.4}\n",
                self.points[i][0],
                self.points[i][1],
                self.points[i][2],
            ))?;
        }

        for i in 0..self.triangles.len() {
            writer.write_fmt(format_args!(
                "3 {} {} {} {} {} {}\n",
                self.triangles[i][0],
                self.triangles[i][1],
                self.triangles[i][2],
                (self.colors[i][0] * 255.0) as u32,
                (self.colors[i][1] * 255.0) as u32,
                (self.colors[i][2] * 255.0) as u32,
            ))?;
        }

        Ok(())
    }
}

impl Debug for Mesh {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Mesh with {} faces / {} points / {} colors",
            self.triangles.len(),
            self.points.len(),
            self.colors.len()
        )
    }
}

fn tetrahedron() -> Mesh {
    let a = 1.0;
    let h_d = (3.0f32.sqrt() / 2.0) * a;
    let h_p = a * ((2.0 / 3.0) as f32).sqrt();
    let center: Vec<f32> = vec![0.0, h_d / 3.0, h_p / 4.0];

    let p1 = [-a / 2.0 - center[0], 0.0 - center[1], 0.0 - center[2]];
    let p2 = [a / 2.0 - center[0], 0.0 - center[1], 0.0 - center[2]];
    let p3 = [0.0 - center[0], h_d - center[1], 0.0 - center[2]];
    let p4 = [0.0 - center[0], h_d / 3.0 - center[1], h_p - center[2]];

    let points: Vec<[f32; 3]> = vec![p1, p2, p3, p4];
    let triangles = vec![[0, 2, 1], [0, 1, 3], [2, 0, 3], [1, 2, 3]];
    let colors = vec![[1.0, 1.0, 1.0]; triangles.len()];
    Mesh::new(points, triangles, colors)
}

fn write_ply_pointcloud(
        mesh: &Mesh,
        filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let header = format!(
        "ply
            format ascii 1.0
            element vertex {num_points}
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            end_header",
        num_points = mesh.points.len(),
    );

    let file = File::create(filename)?;
    let mut writer = BufWriter::new(&file);

    for line in header.lines() {
        writer.write(line.trim().as_bytes())?;
        writer.write("\n".as_bytes())?;
    }

    for i in 0..mesh.points.len() {
        writer.write_fmt(format_args!(
            "{:.4} {:.4} {:.4} {} {} {}\n",
            mesh.points[i][0],
            mesh.points[i][1],
            mesh.points[i][2],
            (mesh.colors[i][0] * 255.0) as u32,
            (mesh.colors[i][1] * 255.0) as u32,
            (mesh.colors[i][2] * 255.0) as u32,
        ))?;
    }

    Ok(())
}


fn main() {

    let mut reader = ImageReader::open("assets/lroc_color_poles.tif").unwrap();
    reader.no_limits();
    let img = reader.decode().unwrap().into_rgb8();

    // let mut reader = ImageReader::open("assets/Lunar_DEM_resized.tif").unwrap();
    // reader.no_limits();
    // let img = reader.decode();
    //
    // println!("{:#?}", img);

    let timer_start = Utc::now();

    let mut mesh = tetrahedron();
    mesh.subdivide(8);
    mesh.project_to_sphere();
    // mesh.fill_colors([1.0, 1.0, 1.0]);
    mesh.add_color(&img);
    mesh.add_seedpoints(1000);

    let pointcloud = Mesh::new(mesh.get_seedpoints(1000), vec![], vec![[0.0, 0.0, 0.0]; 1000]);
    write_ply_pointcloud(&pointcloud, "pointcloud.ply").unwrap();

    println!("{:#?}", mesh);

    mesh.write_ply_with_triangle_colors("test.ply").unwrap();

    let timer_diff = Utc::now() - timer_start;
    println!(
        "Completed in {:.3}s",
        timer_diff.num_milliseconds() as f64 / 1_000.0
    );
}
