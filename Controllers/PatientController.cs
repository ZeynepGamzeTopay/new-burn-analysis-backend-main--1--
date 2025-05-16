using Microsoft.AspNetCore.Mvc;
using BurnAnalysisApp.Data;
using BurnAnalysisApp.Models;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using BurnAnalysisApp.Services;
using System.IO;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http; // IFormFile için gerekli

namespace BurnAnalysisApp.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class PatientController : ControllerBase
    {
        private readonly AppDbContext _context;
        private readonly IConfiguration _configuration;

        public PatientController(AppDbContext context, IConfiguration configuration)
        {
            _context = context;
            _configuration = configuration;
        }

        [HttpGet("search")]
        public async Task<IActionResult> SearchPatients(string searchTerm)
        {
            if (string.IsNullOrWhiteSpace(searchTerm))
                return BadRequest("Arama terimi boş olamaz.");

            var patients = await _context.Patients
                .Where(p => p.Name.Contains(searchTerm))
                .ToListAsync();

            return Ok(patients);
        }

        [HttpGet("{id}")]
        public async Task<IActionResult> GetPatientById(int id)
        {
            var patient = await _context.Patients.FindAsync(id);

            if (patient == null)
                return NotFound("Hasta bulunamadı.");

            return Ok(patient);
        }

        [HttpGet]
        public async Task<IActionResult> GetPatients()
        {
            var patients = await _context.Patients.ToListAsync();
            return Ok(patients);
        }

        [HttpPost]
        public async Task<IActionResult> AddPatient([FromForm] PatientInfo patient, IFormFile photo)
        {
            if (!ModelState.IsValid)
                return BadRequest(ModelState);

            if (photo != null)
            {
                var photoPath = Path.Combine("Uploads", Guid.NewGuid() + Path.GetExtension(photo.FileName));
                using (var stream = new FileStream(photoPath, FileMode.Create))
                {
                    await photo.CopyToAsync(stream);
                }
                patient.PhotoPath = photoPath;
            }

            await _context.Patients.AddAsync(patient);
            await _context.SaveChangesAsync();

            return CreatedAtAction(nameof(GetPatients), new { id = patient.PatientID }, patient);
        }

        [HttpPut("{id}")]
        public async Task<IActionResult> UpdatePatient(int id, [FromBody] PatientInfo patient)
        {
            if (id != patient.PatientID)
                return BadRequest("ID uyuşmazlığı.");

            if (!ModelState.IsValid)
                return BadRequest(ModelState);

            var existingPatient = await _context.Patients.FindAsync(id);
            if (existingPatient == null)
                return NotFound("Hasta bulunamadı.");

            existingPatient.Name = patient.Name;
            existingPatient.Age = patient.Age;
            existingPatient.Gender = patient.Gender;
            existingPatient.MedicalHistory = patient.MedicalHistory;
            existingPatient.BurnCause = patient.BurnCause;
            existingPatient.HospitalArrivalDate = patient.HospitalArrivalDate;
            existingPatient.BurnOccurrenceDate = patient.BurnOccurrenceDate;

            _context.Patients.Update(existingPatient);
            await _context.SaveChangesAsync();

            return NoContent();
        }

        [HttpPut("{id}/burndepth")]
        public async Task<IActionResult> UpdateBurnDepth(int id, [FromBody] string burnDepth)
        {
            var existingPatient = await _context.Patients.FindAsync(id);
            if (existingPatient == null)
                return NotFound("Hasta bulunamadı.");

            existingPatient.BurnDepth = burnDepth;
            _context.Patients.Update(existingPatient);
            await _context.SaveChangesAsync();

            return NoContent();
        }
        public class BurnDepthUpdateDto
        {
            public string? BurnDepth { get; set; }
        }

        [HttpPost("update-burn-depth/{id}")]
        public async Task<IActionResult> UpdateBurnDepthh(int id, [FromBody] BurnDepthUpdateDto dto)
        {
            var patient = await _context.Patients.FindAsync(id);
            if (patient == null) return NotFound();

            patient.BurnDepth = dto.BurnDepth;
            await _context.SaveChangesAsync();
            Console.WriteLine($"Hasta ID: {id}, Yeni Derinlik: {dto.BurnDepth}");
            return Ok(new { message = "Burn depth updated successfully." });
        }

        public class VerifiedUpdateDto
        {
            public bool Verified { get; set; }
        }

        [HttpPost("update-verified/{id}")]
        public async Task<IActionResult> UpdateVerifiedStatus(int id, [FromBody] VerifiedUpdateDto dto)
        {
            var patient = await _context.Patients.FindAsync(id);
            if (patient == null) return NotFound("Hasta bulunamadı.");

            patient.Verified = dto.Verified;
            _context.Patients.Update(patient);
            await _context.SaveChangesAsync();

            return Ok(new { message = $"Verified durumu güncellendi: {dto.Verified}" });
        }
       
        [HttpDelete("{id}")]
        public async Task<IActionResult> DeletePatient(int id)
        {
            var patient = await _context.Patients.FindAsync(id);

            if (patient == null)
                return NotFound("Hasta bulunamadı.");

            if (!string.IsNullOrEmpty(patient.PhotoPath) && System.IO.File.Exists(patient.PhotoPath))
            {
                System.IO.File.Delete(patient.PhotoPath);
            }

            _context.Patients.Remove(patient);
            await _context.SaveChangesAsync();

            return NoContent();
        }

        [HttpPost("{id}/send-reminder")]
        public async Task<IActionResult> SendAppointmentReminder(int id, [FromBody] DateTime appointmentDate)
        {
            var patient = await _context.Patients.FindAsync(id);

            if (patient == null)
                return NotFound("Hasta bulunamadı.");

            if (string.IsNullOrEmpty(patient.Email))
                return BadRequest("Hastanın e-posta adresi bulunamadı.");

            var emailService = new EmailService(_configuration);
            await emailService.SendAppointmentReminderEmailAsync(patient.Email, patient.Name, appointmentDate);

            return Ok("Hatırlatma e-postası başarıyla gönderildi.");
        }

        [HttpPost("upload-audio/{patientId}")]
        public async Task<IActionResult> UploadAudio(IFormFile audio, [FromRoute] int patientId)
        {
            if (audio == null || audio.Length == 0)
            {
                return BadRequest("Ses dosyası yüklenmedi.");
            }

            // Dosya adını ve yolunu oluştur
            var fileName = $"{patientId}_{Guid.NewGuid()}.mp4"; // **.mp4 olarak değiştirildi**
            string webRootPath = _configuration.GetValue<string>("WebRootPath");
            if (string.IsNullOrEmpty(webRootPath))
            {
                webRootPath = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot");
            }

            // Göreli yolu oluştur
            var relativePath = Path.Combine("uploads", "Audio", fileName);

            // Tam yolu oluştur
            var filePath = Path.Combine(webRootPath, "uploads", "Audio", fileName);

            // Dosyayı kaydet
            try
            {
                using (var stream = new FileStream(filePath, FileMode.Create))
                {
                    await audio.CopyToAsync(stream);
                }

                // Veritabanında hasta kaydını güncelle
                var patient = await _context.Patients.FindAsync(patientId);
                if (patient == null)
                {
                    if (System.IO.File.Exists(filePath)) { System.IO.File.Delete(filePath); }
                    return NotFound("Hasta bulunamadı.");
                }

                // Eğer hastanın daha önce ses kaydı varsa, eski kaydı sil
                if (!string.IsNullOrEmpty(patient.AudioPath) && System.IO.File.Exists(patient.AudioPath))
                {
                    System.IO.File.Delete(patient.AudioPath);
                }

                patient.AudioPath = relativePath; // Göreceli yolu kaydet
                _context.Patients.Update(patient);
                await _context.SaveChangesAsync();

                return Ok(new { message = "Ses kaydı başarıyla yüklendi.", filePath });
            }
            catch (Exception ex)
            {
                if (System.IO.File.Exists(filePath))
                {
                    System.IO.File.Delete(filePath);
                }
                return StatusCode(500, $"Ses kaydı yüklenirken bir hata oluştu: {ex.Message}");
            }
        }

        [HttpDelete("delete-audio/{patientId}")]
        public async Task<IActionResult> DeleteAudio(int patientId)
        {
            var patient = await _context.Patients.FindAsync(patientId);
            if (patient == null)
            {
                return NotFound("Hasta bulunamadı.");
            }

            // Eğer hastanın ses kaydı varsa, dosyayı sil
            if (!string.IsNullOrEmpty(patient.AudioPath) && System.IO.File.Exists(patient.AudioPath))
            {
                System.IO.File.Delete(patient.AudioPath);
            }

            patient.AudioPath = null; // Veritabanındaki yolu temizle
            _context.Patients.Update(patient);
            await _context.SaveChangesAsync();

            return NoContent(); // Başarılı silme
        }

    }
}