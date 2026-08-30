/*---------------------------------------------------------------------------
 *
 *  ExaDiS
 *
 *  Nicolas Bertin
 *  bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#include "types.h"
#include "crystal.h"
#include "functions.h"
#include <algorithm>

using namespace ExaDiS;

/*---------------------------------------------------------------------------
 *
 *    Struct:       SystemTest
 *
 *-------------------------------------------------------------------------*/
struct SystemTest {
    int Nnodes;
    Kokkos::View<Vec3*> nodes;
    
    SystemTest() {
        Nnodes = 4;
        Kokkos::resize(nodes, Nnodes);
        auto h_nodes = Kokkos::create_mirror_view(nodes);
        for (int i = 0; i < Nnodes; i++)
            h_nodes(i) = Vec3(0.0);
        Kokkos::deep_copy(nodes, h_nodes);
    }
    
    bool check_results(bool print=0) {
        auto h_nodes = Kokkos::create_mirror_view(nodes);
        Kokkos::deep_copy(h_nodes, nodes);
        if (print) printf(" Results:\n");
        for (int i = 0; i < Nnodes; i++) {
            if ((h_nodes(i)-Vec3(1.0*i)).norm2() > 1e-10) return false;
            if (print) printf("  nodes(%d) = %f %f %f\n", i, h_nodes(i).x, h_nodes(i).y, h_nodes(i).z);
        }
        return true;
    }
};

/*---------------------------------------------------------------------------
 *
 *    Function:     test_system
 *
 *-------------------------------------------------------------------------*/
struct FunctorSystem {
    SystemTest system;
    FunctorSystem(SystemTest& _system) : system(_system) {}
    KOKKOS_INLINE_FUNCTION
    void operator()(const int& i) const {
        system.nodes(i) = Vec3(1.0*i);
    }
};

void test_system()
{
    ExaDiS_log("test_system()\n");
    
    SystemTest* system = new SystemTest();
    try
    {
        Kokkos::parallel_for("FunctorSystem",
            system->Nnodes, FunctorSystem(*system)
        );
        Kokkos::fence();
        ExaDiS_log(" %s\n", system->check_results() ? "PASS" : "FAIL");
    }
    catch(const std::runtime_error& re) {
        ExaDiS_log("Runtime error: %s\n", re.what());
        ExaDiS_log(" FAIL\n");
    } catch(const std::exception& ex) {
        ExaDiS_log("Error occurred: %s\n", ex.what());
        ExaDiS_log(" FAIL\n");
    } catch(...) {
        ExaDiS_log("Unknown error occurred\n");
        ExaDiS_log(" FAIL\n");
    }
    delete system;
}

/*---------------------------------------------------------------------------
 *
 *    Function:     test_system_unified
 *
 *-------------------------------------------------------------------------*/
struct FunctorSystemUnified {
    SystemTest* system;
    FunctorSystemUnified(SystemTest* _system) : system(_system) {}
    KOKKOS_INLINE_FUNCTION
    void operator()(const int& i) const {
        system->nodes(i) = Vec3(1.0*i);
    }
};

void test_system_unified()
{
    ExaDiS_log("test_system_unified()\n");
    
    SystemTest* system = exadis_new<SystemTest>();
    try
    {
        Kokkos::parallel_for("FunctorSystemUnified",
            system->Nnodes, FunctorSystemUnified(system)
        );
        Kokkos::fence();
        ExaDiS_log(" %s\n", system->check_results() ? "PASS" : "FAIL");
    }
    catch(const std::runtime_error& re) {
        ExaDiS_log("Runtime error: %s\n", re.what());
        ExaDiS_log(" FAIL\n");
    } catch(const std::exception& ex) {
        ExaDiS_log("Error occurred: %s\n", ex.what());
        ExaDiS_log(" FAIL\n");
    } catch(...) {
        ExaDiS_log("Unknown error occurred\n");
        ExaDiS_log(" FAIL\n");
    }
    exadis_delete(system);
}

/*---------------------------------------------------------------------------
 *
 *    Function:     test_bcc_plane_families
 *
 *-------------------------------------------------------------------------*/
bool matches_plane_family(Vec3 plane, int family)
{
    double values[3] = {fabs(plane.x), fabs(plane.y), fabs(plane.z)};
    std::sort(values, values+3);
    Vec3 target;
    if (family == 110) target = Vec3(0.0, 1.0, 1.0).normalized();
    else if (family == 112) target = Vec3(1.0, 1.0, 2.0).normalized();
    else if (family == 123) target = Vec3(1.0, 2.0, 3.0).normalized();
    else return false;
    double expected[3] = {fabs(target.x), fabs(target.y), fabs(target.z)};
    std::sort(expected, expected+3);
    for (int i = 0; i < 3; i++)
        if (fabs(values[i]-expected[i]) > 1e-10) return false;
    return true;
}

bool check_bcc_family_mask(int mask, int family, int planes_per_burg)
{
    CrystalParams params;
    params.type = BCC_CRYSTAL;
    params.bcc_plane_family_mask = mask;
    Crystal crystal(params);

    if (!crystal.use_glide_planes || !crystal.enforce_glide_planes) return false;
    if (crystal.num_sys != 4*planes_per_burg) return false;
    for (int i = 0; i < 4; i++) {
        if (crystal.planes_per_burg(i) != planes_per_burg) return false;
        int start = crystal.burg_start_plane(i);
        for (int j = 0; j < planes_per_burg; j++) {
            Vec3 plane = crystal.ref_planes(start+j);
            if (!matches_plane_family(plane, family)) return false;
            if (fabs(dot(plane, crystal.ref_burgs(i))) > 1e-10) return false;
        }

        Vec3 directions[12];
        int count = crystal.get_bcc_screw_glide_directions<SerialDisNet>(
            crystal.ref_burgs(i), directions, 12);
        if (count != planes_per_burg) return false;
        for (int j = 0; j < count; j++) {
            Vec3 plane = cross(crystal.ref_burgs(i), directions[j]).normalized();
            if (!matches_plane_family(plane, family)) return false;
        }
    }

    // Junction Burgers vectors retain all original zonal planes.
    for (int i = 4; i < 7; i++)
        if (crystal.planes_per_burg(i) != 16) return false;
    return true;
}

bool check_bcc_family_selector(int selector, int family, int planes_per_burg)
{
    CrystalParams params;
    params.type = BCC_CRYSTAL;
    params.num_bcc_plane_families = selector;
    Crystal crystal(params);

    if (!crystal.use_glide_planes || !crystal.enforce_glide_planes) return false;
    if (crystal.num_sys != 4*planes_per_burg) return false;
    for (int i = 0; i < 4; i++) {
        if (crystal.planes_per_burg(i) != planes_per_burg) return false;
        int start = crystal.burg_start_plane(i);
        for (int j = 0; j < planes_per_burg; j++)
            if (!matches_plane_family(crystal.ref_planes(start+j), family)) return false;

        Vec3 directions[12];
        int count = crystal.get_bcc_screw_glide_directions<SerialDisNet>(
            crystal.ref_burgs(i), directions, 12);
        if (count != planes_per_burg) return false;
        for (int j = 0; j < count; j++) {
            Vec3 plane = cross(crystal.ref_burgs(i), directions[j]).normalized();
            if (!matches_plane_family(plane, family)) return false;
        }
    }
    return true;
}

bool check_bcc_prismatic_loop(int selector, int family, int expected_sides)
{
    CrystalParams params;
    params.type = BCC_CRYSTAL;
    params.num_bcc_plane_families = selector;
    Crystal crystal(params);
    SerialDisNet network(Cell(100.0));

    Vec3 b = crystal.ref_burgs(0);
    insert_prismatic_loop(crystal, &network, b, 10.0, Vec3(50.0), 20.0);
    if (network.number_of_segs() != expected_sides) return false;
    for (int i = 0; i < network.number_of_segs(); i++) {
        const DisSeg& seg = network.segs[i];
        Vec3 r1 = network.nodes[seg.n1].pos;
        Vec3 r2 = network.cell.pbc_position(r1, network.nodes[seg.n2].pos);
        Vec3 tangent = (r2-r1).normalized();
        if (!matches_plane_family(seg.plane, family)) return false;
        if (fabs(dot(seg.burg.normalized(), seg.plane)) > 1e-10) return false;
        if (fabs(dot(tangent, seg.plane)) > 1e-10) return false;
    }
    return true;
}

void test_bcc_plane_families()
{
    ExaDiS_log("test_bcc_plane_families()\n");

    Crystal legacy(BCC_CRYSTAL);
    bool pass = legacy.num_sys == 12;
    for (int i = 0; i < 4; i++)
        pass = pass && legacy.planes_per_burg(i) == 6;

    pass = pass && check_bcc_family_mask(BCC_PLANE_FAMILY_110, 110, 3);
    pass = pass && check_bcc_family_mask(BCC_PLANE_FAMILY_112, 112, 3);
    pass = pass && check_bcc_family_mask(BCC_PLANE_FAMILY_123, 123, 6);
    pass = pass && check_bcc_family_selector(4, 112, 3);
    pass = pass && check_bcc_family_selector(5, 123, 6);
    pass = pass && check_bcc_prismatic_loop(1, 110, 6);
    pass = pass && check_bcc_prismatic_loop(4, 112, 6);
    pass = pass && check_bcc_prismatic_loop(5, 123, 12);
    ExaDiS_log(" %s\n", pass ? "PASS" : "FAIL");
}

/*---------------------------------------------------------------------------
 *
 *    Function:     main
 *
 *-------------------------------------------------------------------------*/
int main(int argc, char* argv[])
{
    ExaDiS::Initialize init(argc, argv);
    
    std::string test_name = "";
    if (argc > 1)
        test_name = std::string(argv[1]);
    
    if (test_name == "test_system" || test_name.empty())
        test_system();
    if (test_name == "test_system_unified" || test_name.empty())
        test_system_unified();
    if (test_name == "test_bcc_plane_families" || test_name.empty())
        test_bcc_plane_families();
    
    return 0;
}
