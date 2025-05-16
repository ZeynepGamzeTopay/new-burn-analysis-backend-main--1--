using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using BurnAnalysisApp.Data;
using BurnAnalysisApp.Models;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using System.IO;

namespace BurnAnalysisApp.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class VisitController : ControllerBase
    {
        private readonly AppDbContext _context;

        public VisitController(AppDbContext context)
        {
            _context = context;
        }

        // Tüm ziyaretleri getir
        [HttpGet]
        public async Task<ActionResult<IEnumerable<Visit>>> GetVisits()
        {
            return await _context.Visits.Include(v => v.Patient).ToListAsync();
        }

        // ID'ye göre tek bir ziyareti getir
        [HttpGet("{id}")]
        public async Task<ActionResult<Visit>> GetVisitById(int id)
        {
            var visit = await _context.Visits.Include(v => v.Patient)
                                             .FirstOrDefaultAsync(v => v.VisitID == id);

            if (visit == null)
            {
                return NotFound(new { message = "Visit not found" });
            }

            return visit;
        }

        
   // Yeni bir ziyaret oluştur
[HttpPost("patient/{patientId}")]
public async Task<ActionResult<Visit>> CreateVisit(int patientId, [FromBody] Visit visit)
{
    if (visit == null)
    {
        return BadRequest(new { message = "Veri null, geçersiz ziyaret." });
    }

    // Hasta ID'yi ziyaret nesnesine set et
    visit.PatientID = patientId;  // Burada PatientID kullanıyoruz

    try
    {
        _context.Visits.Add(visit);
        await _context.SaveChangesAsync();
    }
    catch (Exception ex)
    {
        return BadRequest(new { message = $"Bir hata oluştu: {ex.Message}" });
    }

    return CreatedAtAction(nameof(GetVisitById), new { id = visit.VisitID }, visit);
}



        // Mevcut bir ziyareti güncelle
        [HttpPut("{id}")]
        public async Task<IActionResult> UpdateVisit(int id, [FromBody] Visit visit)
        {
            if (id != visit.VisitID)
            {
                return BadRequest(new { message = "Visit ID eşleşmiyor" });
            }

            var existingVisit = await _context.Visits.FindAsync(id);
            if (existingVisit == null)
            {
                return NotFound(new { message = "Visit bulunamadı" });
            }

            existingVisit.VisitDate = visit.VisitDate;
            existingVisit.PhotoPath = visit.PhotoPath;
            existingVisit.LabResultsFilePath = visit.LabResultsFilePath;
            existingVisit.PrescribedMedications = visit.PrescribedMedications;
            existingVisit.Notes = visit.Notes;

            _context.Entry(existingVisit).State = EntityState.Modified;
            await _context.SaveChangesAsync();

            return NoContent();
        }

        // Ziyareti sil
        [HttpDelete("{id}")]
        public async Task<IActionResult> DeleteVisit(int id)
        {
            var visit = await _context.Visits.FindAsync(id);
            if (visit == null)
            {
                return NotFound(new { message = "Visit not found" });
            }

            _context.Visits.Remove(visit);
            await _context.SaveChangesAsync();

            return NoContent();
        }

        // Belirli bir hastaya ait ziyaretleri getir
        [HttpGet("patient/{patientId}")]
        public async Task<ActionResult<IEnumerable<Visit>>> GetVisitsByPatientId(int patientId)
        {
            if (patientId <= 0)
            {
                return BadRequest(new { message = "Geçersiz hasta ID" });
            }

            var visits = await _context.Visits
                                       .Where(v => v.PatientID == patientId)
                                       .ToListAsync();

            return Ok(visits);
        }

        [HttpPost("upload")]
        public async Task<IActionResult> UploadVisitFiles([FromForm] IFormFile file)
        {
            if (file == null || file.Length == 0)
            {
                return BadRequest(new { message = "Dosya seçilmedi" });
            }

            var uploadsFolder = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot/uploads");
            if (!Directory.Exists(uploadsFolder))
            {
                Directory.CreateDirectory(uploadsFolder);
            }

            var fileName = Guid.NewGuid().ToString() + Path.GetExtension(file.FileName);
            var filePath = Path.Combine(uploadsFolder, fileName);

            using (var stream = new FileStream(filePath, FileMode.Create))
            {
                await file.CopyToAsync(stream);
            }

            return Ok(new { filePath = $"/uploads/{fileName}" });
        }
        
    }
}