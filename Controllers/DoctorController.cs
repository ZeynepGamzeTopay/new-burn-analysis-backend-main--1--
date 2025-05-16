using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using BurnAnalysisApp.Data;
using BurnAnalysisApp.Models;
using BurnAnalysisApp.Services;
using Microsoft.AspNetCore.Authorization;
using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Security.Cryptography; // SHA256 için gerekli
using System.Text; // Encoding için gerekli

namespace BurnAnalysisApp.Controllers
{
    [ApiController]
    [Route("api/doctor")]
    public class DoctorController : ControllerBase
    {
        private readonly AppDbContext _context;
        private readonly IEmailService _emailService;
        private readonly IJwtService _jwtService;

        public DoctorController(AppDbContext context, IEmailService emailService, IJwtService jwtService)
        {
            _context = context;
            _emailService = emailService;
            _jwtService = jwtService;
        }

        [HttpPost("login")]
public async Task<IActionResult> Login([FromBody] Doctor loginRequest)
{
    if (string.IsNullOrWhiteSpace(loginRequest.Email) || string.IsNullOrWhiteSpace(loginRequest.Password))
        return BadRequest(new { success = false, message = "Email and Password are required." });

    // Girilen şifreyi hashle
    var hashedPassword = HashPassword(loginRequest.Password);

    var doctor = await _context.Doctors
        .FirstOrDefaultAsync(d => d.Email == loginRequest.Email && d.Password == hashedPassword);

    if (doctor == null)
        return Unauthorized(new { success = false, message = "Invalid credentials." });

    if (!doctor.Verified)
        return Unauthorized(new { success = false, message = "Your account has not been verified yet." });

    // Token oluştur
    var token = _jwtService.GenerateToken(doctor.DoctorID, doctor.Email);

    return Ok(new { success = true, message = "Login successful.", token });
}

// Şifreyi SHA256 ile hashleme fonksiyonunu ekle
private string HashPassword(string password)
{
    using (var sha256 = SHA256.Create())
    {
        var bytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(password));
        return Convert.ToBase64String(bytes);
    }
}


        [HttpGet]
        public async Task<IActionResult> GetAllDoctors()
        {
            var doctors = await _context.Doctors.ToListAsync();
            return Ok(doctors);
        }

      [Authorize]
[HttpGet("info")]
public async Task<IActionResult> GetDoctorInfo()
{
    var doctorId = GetDoctorIdFromToken();
    if (doctorId == null)
        return Unauthorized(new { success = false, message = "Invalid token." });

    var doctor = await _context.Doctors
        .Where(d => d.DoctorID == doctorId)
        .Select(d => new {
            d.DoctorID,
            d.Name,
            d.Email,
            Password = "" // Şifreyi boş döndür, istemciye şifreyi göstermemek için
        })
        .FirstOrDefaultAsync();

    if (doctor == null)
        return NotFound(new { success = false, message = "Doctor not found." });

    return Ok(doctor);
}

        [HttpPut("{id}")]
        public async Task<IActionResult> UpdateDoctor(int id, [FromBody] Doctor updatedDoctor)
        {
            var doctor = await _context.Doctors.FindAsync(id);
            if (doctor == null) return NotFound();

            var wasVerified = doctor.Verified;

            doctor.Name = updatedDoctor.Name;
            doctor.Email = updatedDoctor.Email;
            doctor.AdminID = updatedDoctor.AdminID;
            doctor.Verified = updatedDoctor.Verified;

            if (!wasVerified && doctor.Verified)
            {
                try
                {
                    await _emailService.SendVerificationEmailAsync(doctor.Email, doctor.Name);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error sending verification email: {ex.Message}");
                }
            }

            await _context.SaveChangesAsync();

            return NoContent();
        }

        [HttpDelete("{id}")]
        public async Task<IActionResult> DeleteDoctor(int id)
        {
            var doctor = await _context.Doctors.FindAsync(id);
            if (doctor == null) return NotFound();

            _context.Doctors.Remove(doctor);
            await _context.SaveChangesAsync();
            return NoContent();
        }

        private int? GetDoctorIdFromToken()
        {
            var identity = HttpContext.User.Identity as ClaimsIdentity;
            if (identity == null)
                return null;

            var doctorIdClaim = identity.FindFirst(ClaimTypes.NameIdentifier);
            return doctorIdClaim != null ? int.Parse(doctorIdClaim.Value) : (int?)null;
        }

        [HttpGet("status")]
        public async Task<IActionResult> GetDoctorsByStatus([FromQuery] string status)
        {
            IQueryable<Doctor> doctorsQuery = _context.Doctors;

            if (!string.IsNullOrEmpty(status))
            {
                if (status == "pending")
                {
                    doctorsQuery = doctorsQuery.Where(d => !d.Verified);
                }
                else if (status == "approved")
                {
                    doctorsQuery = doctorsQuery.Where(d => d.Verified);
                }
            }

            var doctors = await doctorsQuery.ToListAsync();
            return Ok(doctors);
        }
[Authorize]
[HttpPut("update/{id}")]
public async Task<IActionResult> UpdateDoctorInfo(int id, [FromBody] Doctor updatedDoctor)


{
    var doctorId = GetDoctorIdFromToken();
    if (doctorId == null)
        return Unauthorized(new { success = false, message = "Invalid token." });

    var doctor = await _context.Doctors.FindAsync(doctorId);
    if (doctor == null)
        return NotFound(new { success = false, message = "Doctor not found." });

    doctor.Name = updatedDoctor.Name;
    doctor.Email = updatedDoctor.Email;

    // Eğer frontend boş password gönderirse eski şifreyi koru
    if (!string.IsNullOrWhiteSpace(updatedDoctor.Password))
    {
        doctor.Password = updatedDoctor.Password;
    }

    await _context.SaveChangesAsync();

    return Ok(new { success = true, message = "Profile updated successfully." });
}

[HttpGet("assigned/{adminId}")]
public async Task<IActionResult> GetDoctorsByAdmin(int adminId)
{
    var doctors = await _context.Doctors
        .Where(d => d.AdminID == adminId)
        .ToListAsync();

    if (doctors == null || doctors.Count == 0)
    {
        return NotFound(new { message = "Bu admin'e atanmış doktor bulunamadı." });
    }

    return Ok(doctors);
}


    }

}
